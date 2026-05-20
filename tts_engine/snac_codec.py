# tts_engine/snac_codec.py
from __future__ import annotations
import asyncio
import logging
import threading
from snac import SNAC
from typing import List, Optional
import numpy as np
import torch
from .utils import resample_audio
from .timing import track_time

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading SNAC model for each instance
_SNAC_MODEL_CACHE: dict[str, SNAC] = {}


def _get_or_load_snac_model(device: str, model_name: str = "hubertsiuzdak/snac_24khz") -> SNAC:
    """Get cached SNAC model or load it if not cached.

    Models are cached per (model_name, device) so multiple SNACCodec instances
    share the same nn.Module.
    """
    cache_key = f"{model_name}_{device}"

    if cache_key not in _SNAC_MODEL_CACHE:
        logger.info("Loading SNAC model %s on device %s", model_name, device)
        model = SNAC.from_pretrained(model_name).eval().to(device)
        _SNAC_MODEL_CACHE[cache_key] = model
    else:
        logger.debug("Using cached SNAC model: %s", cache_key)

    return _SNAC_MODEL_CACHE[cache_key]


class SNACCodec:
    """
    Unified SNAC codec for encoding audio to tokens and decoding tokens to audio.
    
    Supports both:
    - Encoding: audio waveform → SNAC tokens (for zero-shot voice cloning)
    - Decoding: SNAC tokens → PCM16 audio (for TTS synthesis)
    
    Uses a global model cache to avoid reloading the SNAC model when creating
    multiple instances, which significantly improves initialization time.
    """
    
    def __init__(self, device: Optional[str] = None, model_name: str = "hubertsiuzdak/snac_24khz"):
        """
        Initialize SNAC codec.
        
        Args:
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
            model_name: HuggingFace model identifier for SNAC
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.model_name = model_name
        self.sample_rate = 24000  # SNAC 24kHz model

        # Get or load model from cache
        self.model = _get_or_load_snac_model(device, model_name)

        # Concurrency guard: the SNAC nn.Module is shared across requests but its
        # forward is not safe to invoke from multiple coroutines simultaneously on
        # the same default CUDA stream. The lock serializes Python-side entry; the
        # dedicated CUDA stream keeps SNAC work off vLLM's stream.
        self._async_lock = asyncio.Lock()
        self._thread_lock = threading.Lock()
        self._stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=device) if device == "cuda" else None
        )

        # torch.compile the decode path. SNAC decode is called O(100×) per
        # request; reduce-overhead mode minimizes per-call dispatch cost
        # (~10ms saved per window after warmup). Shapes are static (28-code
        # windows always), so dynamic=False is correct.
        self._compiled_decode = self.model.decode
        if device == "cuda" and hasattr(torch, "compile"):
            try:
                self._compiled_decode = torch.compile(
                    self.model.decode, mode="reduce-overhead", dynamic=False
                )
                logger.info("torch.compile(SNAC.decode) enabled")
            except Exception as e:
                logger.warning("torch.compile(SNAC.decode) failed: %s; using eager", e)

    @track_time("SNAC.encode_audio")
    def encode_audio(
        self,
        audio: torch.Tensor,
        input_sample_rate: int = 24000,
        add_token_offsets: bool = True
    ) -> List[int]:
        """
        Encode audio waveform to SNAC tokens for zero-shot voice cloning.
        
        This method takes audio and converts it to a sequence of tokens that
        can be used as a voice reference in the Svara-TTS model.
        Automatically resamples audio to 24kHz if needed.
        
        Args:
            audio: Audio tensor of shape (channels, samples) or (samples,).
                   If 1D, will be converted to (1, 1, samples) for SNAC.
            input_sample_rate: Sample rate of the input audio in Hz. If not 24000,
                              audio will be automatically resampled to 24kHz.
            add_token_offsets: If True, adds Svara-TTS token offsets (128266 + vocab offsets)
                              to make tokens ready for model input. If False, returns raw
                              SNAC codes in range [0, 4096].
        
        Returns:
            List of token IDs. Length will be 7 * num_frames where num_frames depends
            on input audio length. Each frame represents ~10ms of audio.
        
        Example:
            >>> codec = SNACCodec()
            >>> # Load 1 second of audio at 48kHz - will be resampled to 24kHz
            >>> audio = torch.randn(48000)
            >>> tokens = codec.encode_audio(audio, input_sample_rate=48000)
            >>> len(tokens) # ~7 tokens per frame, ~100 frames per second
            700
        """
        # Resample to 24kHz if needed
        if input_sample_rate != self.sample_rate:
            audio = resample_audio(audio, input_sample_rate, self.sample_rate, self.device)

        # Ensure proper shape: SNAC expects (batch, channels, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)

        # Move to device and ensure float32
        audio = audio.to(dtype=torch.float32, device=self.device)

        # Encode with SNAC
        with torch.inference_mode():
            codes = self.model.encode(audio)
        
        # SNAC produces hierarchical codes with different temporal resolutions:
        # codes[0]: coarsest (e.g., 100 frames for 1 sec)
        # codes[1]: 2x finer (e.g., 200 frames)
        # codes[2]: 4x finer (e.g., 400 frames)
        # 
        # Interleave pattern per finest-resolution frame:
        # For every 4 frames in codes[2], we get:
        # - 1 code from codes[0]
        # - 2 codes from codes[1] 
        # - 4 codes from codes[2]
        # Output order per coarse frame: c0, c1, c2, c3, c4, c5, c6
        
        all_codes = []
        num_coarse_frames = codes[0].shape[1]
        
        for i in range(num_coarse_frames):
            # Get indices for hierarchical codes
            # Each coarse frame i corresponds to:
            # - codes[0][i]
            # - codes[1][2*i : 2*i+2] (2 codes)
            # - codes[2][4*i : 4*i+4] (4 codes)
            
            c0 = codes[0][0][i].item()
            c1 = codes[1][0][2 * i].item()
            c2 = codes[2][0][4 * i].item()
            c3 = codes[2][0][4 * i + 1].item()
            c4 = codes[1][0][2 * i + 1].item()
            c5 = codes[2][0][4 * i + 2].item()
            c6 = codes[2][0][4 * i + 3].item()
            
            if add_token_offsets:
                # Add Svara-TTS vocabulary offsets
                all_codes.append(c0 + 128266)
                all_codes.append(c1 + 128266 + 4096)
                all_codes.append(c2 + 128266 + (2 * 4096))
                all_codes.append(c3 + 128266 + (3 * 4096))
                all_codes.append(c4 + 128266 + (4 * 4096))
                all_codes.append(c5 + 128266 + (5 * 4096))
                all_codes.append(c6 + 128266 + (6 * 4096))
            else:
                # Raw SNAC codes
                all_codes.extend([c0, c1, c2, c3, c4, c5, c6])
        
        return all_codes
    
    @track_time("SNAC.decode_window")
    def decode_window(self, window: List[int]) -> bytes:
        """Synchronous decode of a sliding window of Svara-TTS codes into PCM16 bytes.

        Window: flat list of raw SNAC codes (no token offsets) in [0, 4096], length
        multiple of 7. Returns PCM16 mono bytes, or b"" for invalid input.

        Safe to call from multiple threads — protected by a process-wide thread
        lock and routed to a dedicated CUDA stream so it doesn't interleave with
        vLLM's default stream.
        """
        with self._thread_lock:
            return self._decode_window_locked(window)

    async def decode_window_async(self, window: List[int]) -> bytes:
        """Async wrapper: serialize via asyncio.Lock, run the GPU kernel in a thread."""
        async with self._async_lock:
            return await asyncio.to_thread(self._decode_window_locked, window)

    def _decode_window_locked(self, window: List[int]) -> bytes:
        if not window or len(window) < 7:
            return b""

        F = len(window) // 7
        frame = window[: F * 7]

        t = torch.tensor(frame, dtype=torch.int32, device=self.device)
        t = t.view(F, 7)

        codes_0 = t[:, 0].reshape(1, -1)
        codes_1 = t[:, [1, 4]].reshape(1, -1)
        codes_2 = t[:, [2, 3, 5, 6]].reshape(1, -1)

        if (
            torch.any((codes_0 < 0) | (codes_0 > 4096))
            or torch.any((codes_1 < 0) | (codes_1 > 4096))
            or torch.any((codes_2 < 0) | (codes_2 > 4096))
        ):
            return b""

        if self._stream is not None:
            with torch.cuda.stream(self._stream), torch.inference_mode():
                audio = self._compiled_decode([codes_0, codes_1, codes_2])
                audio = audio[:, :, 2048:4096]
            self._stream.synchronize()
        else:
            with torch.inference_mode():
                audio = self._compiled_decode([codes_0, codes_1, codes_2])
                audio = audio[:, :, 2048:4096]

        x = audio.detach().float().cpu().numpy().reshape(-1)
        pcm16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
        return pcm16.tobytes()

    def warmup(self) -> None:
        """Run dummy decodes to JIT-compile and CUDA-graph-capture before serving.

        With torch.compile(mode='reduce-overhead'), the first call traces and
        captures; the second call replays the graph. Doing both up front means
        the first real request doesn't pay either cost.
        """
        dummy = [100] * 28  # valid SNAC code range, four frames
        # First call: compile + graph capture. May fall back to eager on error.
        try:
            self._decode_window_locked(dummy)
        except Exception as e:
            logger.warning(
                "torch.compile path errored during warmup: %s. Falling back to eager.", e
            )
            self._compiled_decode = self.model.decode
            self._decode_window_locked(dummy)
        # Second call: replay the captured graph so CUDA caches are warm.
        self._decode_window_locked(dummy)
        if self.device == "cuda":
            torch.cuda.synchronize()
        logger.info("SNAC warmup complete on %s", self.device)