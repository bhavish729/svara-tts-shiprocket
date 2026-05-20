
from __future__ import annotations
from typing import Iterator, AsyncIterator, List, Optional, Literal, Union
import concurrent.futures
import asyncio
import logging
from .transports import VLLMCompletionsTransport, VLLMCompletionsTransportAsync
from .mapper import SvaraMapper, extract_custom_token_numbers
from .snac_codec import SNACCodec
from .utils import svara_prompt, create_speaker_id
from .buffers import AudioBuffer, SyncFuture
from .timing import track_time

logger = logging.getLogger(__name__)

class SvaraTTSOrchestrator:
    """
    Sync/Async TTS orchestrator:
    transport -> mapper -> decoder -> PCM int16 chunks.
    
    Args:
        base_url: The base URL of the VLLM server.
        model: The model name.
        speaker_id: The speaker identifier (e.g., "Hindi (Male)", "English (Female)").
                    If not provided, will be constructed from lang_code and gender.
        lang_code: An ISO 639-1 language code (used if speaker_id not provided).
        gender: The gender of the voice (used if speaker_id not provided).
        headers: The headers for the VLLM server.
        prebuffer_seconds: The number of seconds to prebuffer before yielding audio.
        concurrent_decode: If True, decode concurrently.
        max_workers: The number of workers to use for decoding.
        device: Device for SNAC decoder (cuda, mps, cpu, or None for auto).
    """
    def __init__(self,
                 base_url: str,
                 model: str = "kenpath/svara-tts-v1",
                 speaker_id: Optional[str] = None,
                 lang_code: str = "en",
                 gender: Literal["male", "female"] = "male",
                 headers: Optional[dict] = None,
                 prebuffer_seconds: float = 0.5,
                 concurrent_decode: bool = True,
                 max_workers: int = 2,
                 device: Optional[str] = None):
        # If speaker_id is provided, use it; otherwise construct from lang_code and gender
        if speaker_id is None:
            self.speaker_id = create_speaker_id(lang_code, gender)
        else:
            self.speaker_id = speaker_id
        
        self.transport = VLLMCompletionsTransport(base_url, model, headers)
        # Eager-init the async transport so the first concurrent request doesn't
        # race on lazy construction.
        async_base_url = self.transport.url[:-len("/completions")]
        self.transport_async = VLLMCompletionsTransportAsync(
            async_base_url, model, self.transport.headers
        )
        # NOTE: mapper is intentionally NOT instantiated here. Each request gets
        # its own SvaraMapper inside _stream_one/_astream_one — sharing one
        # mapper across concurrent requests interleaves their code streams and
        # produces empty windows.
        self.codec = SNACCodec(device)
        self.prebuffer_samples = int(self.codec.sample_rate * prebuffer_seconds)
        self.concurrent_decode = concurrent_decode
        self.max_workers = max_workers
        # Single shared executor for the sync decode path. The async path uses
        # asyncio.to_thread inside SNACCodec.decode_window_async and doesn't
        # touch this executor.
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="svara-decode"
            )
            if concurrent_decode
            else None
        )
        
    # ------------ SYNC path ------------
    def stream(self, text: str, prompt: Optional[Union[str, List[int]]] = None, **gen_kwargs) -> Iterator[bytes]:
        """Stream the TTS output.
        
        Args:
            text: The text to synthesize.
            prompt: Optional pre-computed prompt. Can be a string or list of token IDs.
                   If None, builds prompt from text and speaker_id.
            gen_kwargs: Additional keyword arguments to pass to the transport.
        """
        yield from self._stream_one(text, prompt=prompt, **gen_kwargs)

    @track_time("Orchestrator.stream_one")
    def _stream_one(self, text: str, prompt: Optional[Union[str, List[int]]] = None, **gen_kwargs) -> Iterator[bytes]:
        # Use provided prompt or build from text + speaker_id
        if prompt is None:
            prompt = svara_prompt(text, self.speaker_id)
        
        # Log prompt details
        if isinstance(prompt, list):
            logger.info(f"Final prompt before inference: {len(prompt)} token IDs")
            logger.debug(f"Token IDs (first 50): {prompt[:50]}")
        else:
            logger.info(f"Final prompt before tokenization: {len(prompt)} chars")
            logger.debug(f"Full prompt: {prompt}")
        
        audio_buf = AudioBuffer(self.prebuffer_samples)
        mapper = SvaraMapper()
        pending: List[concurrent.futures.Future] = []

        def decode(win: List[int]) -> bytes:
            return self.codec.decode_window(win)

        def submit(win: List[int]):
            if self._executor is not None:
                return self._executor.submit(decode, win)
            return SyncFuture(decode(win))

        for token_text in self.transport.stream(prompt, **gen_kwargs):
            for n in extract_custom_token_numbers(token_text):
                win = mapper.feed_raw(n)
                if win is not None:
                    pending.append(submit(win))

                while len(pending) > 2:
                    result = audio_buf.process(pending.pop(0).result())
                    if result:
                        yield result

        for fut in pending:
            result = audio_buf.process(fut.result())
            if result:
                yield result

    # ------------ ASYNC path ------------
    async def astream(
        self,
        text: str,
        prompt: Optional[Union[str, List[int]]] = None,
        *,
        request_id: Optional[str] = None,
        **gen_kwargs,
    ) -> AsyncIterator[bytes]:
        """Async stream the TTS output.

        Args:
            text: Text to synthesize.
            prompt: Optional pre-computed prompt. String or list of token IDs.
                If None, builds prompt from text and speaker_id.
            request_id: Per-request correlation id passed through to vLLM.
            gen_kwargs: Forwarded to the transport.
        """
        async for b in self._astream_one(text, prompt=prompt, request_id=request_id, **gen_kwargs):
            yield b

    @track_time("Orchestrator.astream_one")
    async def _astream_one(
        self,
        text: str,
        prompt: Optional[Union[str, List[int]]] = None,
        *,
        request_id: Optional[str] = None,
        **gen_kwargs,
    ) -> AsyncIterator[bytes]:
        if prompt is None:
            prompt = svara_prompt(text, self.speaker_id)

        if isinstance(prompt, list):
            logger.info("Prompt (request_id=%s): %d token IDs", request_id, len(prompt))
        else:
            logger.info("Prompt (request_id=%s): %d chars", request_id, len(prompt))

        # Per-request state — never shared across concurrent requests.
        audio_buf = AudioBuffer(self.prebuffer_samples)
        mapper = SvaraMapper()
        pending: List[asyncio.Task] = []

        async for token_text in self.transport_async.astream(
            prompt, request_id=request_id, **gen_kwargs
        ):
            for n in extract_custom_token_numbers(token_text):
                win = mapper.feed_raw(n)
                if win is not None:
                    pending.append(asyncio.create_task(self.codec.decode_window_async(win)))

                while len(pending) > 2:
                    result = audio_buf.process(await pending.pop(0))
                    if result:
                        yield result

        for task in pending:
            result = audio_buf.process(await task)
            if result:
                yield result

    async def close(self) -> None:
        """Release the decode executor and the shared transport session."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        from .transports import close_session
        await close_session()
