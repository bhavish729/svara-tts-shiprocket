
from __future__ import annotations
import json
import logging
from typing import Iterator, AsyncIterator, Optional, Dict, Any, Union, List
import requests
import aiohttp
from .timing import track_time

logger = logging.getLogger(__name__)


# Module-level shared aiohttp session.
# vLLM SSE streams are long-lived; a per-request ClientSession (the previous
# default) thrashes the connector pool under 50+ concurrent requests. One
# pooled session with a sized TCPConnector keeps SNAC's pipeline fed.
_session: Optional[aiohttp.ClientSession] = None


def get_session() -> aiohttp.ClientSession:
    """Return the process-wide aiohttp session, building it on first use.

    Must be called from an async context (we need a running loop to bind the
    session's connector to). The session has no total timeout because streaming
    responses are open-ended; sock_connect/sock_read protect against silent
    hangs at the boundaries.
    """
    global _session
    if _session is None or _session.closed:
        connector = aiohttp.TCPConnector(
            limit=128,
            limit_per_host=128,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=300)
        _session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    return _session


async def close_session() -> None:
    """Close the shared session. Call from FastAPI lifespan shutdown."""
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
        _session = None
class VLLMCompletionsTransport:
    """
    Sync transport for OpenAI-compatible /v1/completions streaming (SSE).
    Yields incremental text from choices[0].text.
    """
    def __init__(self, base_url: str, model: str, headers: Optional[Dict[str, str]] = None):
        self.url = base_url.rstrip('/') + '/completions'
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        if headers:
            self.headers.update(headers)

    @track_time("vLLM.stream")
    def stream(self, prompt: Union[str, List[int]], **gen_kwargs) -> Iterator[str]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "stream": True,
            "max_tokens": gen_kwargs.get("max_tokens", 2048),
            "temperature": gen_kwargs.get("temperature", 0.75),
            "top_p": gen_kwargs.get("top_p", 0.9),
            "do_sample": gen_kwargs.get("do_sample", True),
            "repetition_penalty": gen_kwargs.get("repetition_penalty", 1.1),
            "stop_token_ids": [128258,128262]
        }
        
        # Use prompt_token_ids if we have a list, otherwise use prompt string
        if isinstance(prompt, list):
            payload["prompt_token_ids"] = prompt
        else:
            payload["prompt"] = prompt

        with requests.post(self.url, headers=self.headers, json=payload, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                text = choices[0].get("text") or ""
                if text:
                    yield text


class VLLMCompletionsTransportAsync:
    """
    Async transport for OpenAI-compatible /v1/completions streaming (SSE).
    Requires aiohttp.
    """
    def __init__(self, base_url: str, model: str, headers: Optional[Dict[str, str]] = None):
        if aiohttp is None:
            raise RuntimeError("aiohttp is not installed. pip install aiohttp")
        self.url = base_url.rstrip('/') + '/completions'
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        if headers:
            self.headers.update(headers)

    @track_time("vLLM.astream")
    async def astream(
        self,
        prompt: Union[str, List[int]],
        *,
        request_id: Optional[str] = None,
        **gen_kwargs,
    ) -> AsyncIterator[str]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "stream": True,
            "max_tokens": gen_kwargs.get("max_tokens", 2048),
            "temperature": gen_kwargs.get("temperature", 0.75),
            "top_p": gen_kwargs.get("top_p", 0.9),
            "repetition_penalty": gen_kwargs.get("repetition_penalty", 1.1),
            "stop_token_ids": [128258, 128262],
        }

        # Use prompt_token_ids if we have a list, otherwise use prompt string
        if isinstance(prompt, list):
            payload["prompt_token_ids"] = prompt
        else:
            payload["prompt"] = prompt

        # vLLM's OpenAI-compat endpoint accepts a request_id field and surfaces
        # it in logs/metrics — gives us per-request correlation under load.
        if request_id is not None:
            payload["request_id"] = request_id

        sess = get_session()
        async with sess.post(self.url, headers=self.headers, json=payload) as resp:
            if resp.status != 200:
                error_body = await resp.text()
                logger.error(
                    "vLLM request failed [request_id=%s status=%s]: %s",
                    request_id, resp.status, error_body,
                )
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                text = choices[0].get("text") or ""
                if text:
                    yield text
