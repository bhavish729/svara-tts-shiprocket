#!/usr/bin/env python3
"""Concurrent load benchmark for Svara TTS streaming endpoint.

Fires N requests in parallel against /v1/text-to-speech with stream=true and
asserts that every request returns non-empty PCM audio of at least
MIN_BYTES_PER_REQUEST. Reports TTFB and total-bytes percentiles plus a
pass/fail count.

Usage:
    python scripts/bench_concurrent.py --concurrency 50
    python scripts/bench_concurrent.py --concurrency 50 --url http://localhost:8080
    python scripts/bench_concurrent.py --sweep 1,5,10,25,50,100

The 50-concurrent run is the gate before pushing to RunPod: all 50 must return
non-empty audio.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import aiohttp


DEFAULT_TEXT = (
    "The quick brown fox jumps over the lazy dog. Pack my box with five "
    "dozen liquor jugs. Sphinx of black quartz, judge my vow."
)
DEFAULT_VOICE = "English (Male)"
# 24 kHz mono int16 = 48000 bytes/sec. The default text is ~7s of audio, so
# anything under 20 KB indicates the response was truncated or empty.
MIN_BYTES_PER_REQUEST = 20_000


@dataclass
class Result:
    idx: int
    ttfb_s: Optional[float]
    total_s: float
    bytes_received: int
    status: Optional[int]
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return (
            self.error is None
            and self.status == 200
            and self.bytes_received >= MIN_BYTES_PER_REQUEST
        )


async def one_request(
    session: aiohttp.ClientSession,
    url: str,
    text: str,
    voice: str,
    idx: int,
    max_tokens: Optional[int],
) -> Result:
    payload = {
        "text": text,
        "voice": voice,
        "stream": True,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    t0 = time.perf_counter()
    ttfb_s: Optional[float] = None
    total_bytes = 0
    status: Optional[int] = None
    error: Optional[str] = None

    try:
        async with session.post(url, json=payload) as resp:
            status = resp.status
            if status != 200:
                body = await resp.text()
                error = f"HTTP {status}: {body[:200]}"
                return Result(idx, None, time.perf_counter() - t0, 0, status, error)

            async for chunk in resp.content.iter_any():
                if not chunk:
                    continue
                if ttfb_s is None:
                    ttfb_s = time.perf_counter() - t0
                total_bytes += len(chunk)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    return Result(idx, ttfb_s, time.perf_counter() - t0, total_bytes, status, error)


def percentiles(xs: List[float]) -> str:
    if not xs:
        return "n/a"
    xs_sorted = sorted(xs)
    def pct(p: float) -> float:
        k = max(0, min(len(xs_sorted) - 1, int(round(p * (len(xs_sorted) - 1)))))
        return xs_sorted[k]
    return (
        f"p50={pct(0.50)*1000:6.1f}ms  "
        f"p95={pct(0.95)*1000:6.1f}ms  "
        f"p99={pct(0.99)*1000:6.1f}ms  "
        f"max={max(xs_sorted)*1000:6.1f}ms"
    )


async def run_once(
    base_url: str,
    concurrency: int,
    text: str,
    voice: str,
    max_tokens: Optional[int],
) -> List[Result]:
    url = base_url.rstrip("/") + "/v1/text-to-speech"
    connector = aiohttp.TCPConnector(limit=concurrency * 2, limit_per_host=concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=180)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            asyncio.create_task(one_request(session, url, text, voice, i, max_tokens))
            for i in range(concurrency)
        ]
        return await asyncio.gather(*tasks)


def report(label: str, results: List[Result]) -> bool:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    print(f"\n=== {label} ===")
    print(f"  pass: {len(ok)}/{len(results)}")
    if failed:
        print(f"  fail: {len(failed)}")
        # Group failure reasons
        reasons: dict[str, int] = {}
        for r in failed:
            if r.error:
                key = r.error.split("\n")[0][:80]
            elif r.status != 200:
                key = f"HTTP {r.status}"
            else:
                key = f"short: {r.bytes_received}B < {MIN_BYTES_PER_REQUEST}B"
            reasons[key] = reasons.get(key, 0) + 1
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {v:3d}× {k}")

    ttfbs = [r.ttfb_s for r in results if r.ttfb_s is not None]
    print(f"  TTFB:  {percentiles(ttfbs)}")
    totals = [r.total_s for r in results]
    print(f"  total: {percentiles(totals)}")
    sizes = [r.bytes_received for r in results]
    if sizes:
        print(
            f"  bytes: min={min(sizes):>7d}  "
            f"median={int(statistics.median(sizes)):>7d}  "
            f"max={max(sizes):>7d}"
        )
    return len(ok) == len(results)


async def main_async(args: argparse.Namespace) -> int:
    if args.sweep:
        steps = [int(s) for s in args.sweep.split(",")]
        all_passed = True
        for n in steps:
            results = await run_once(args.url, n, args.text, args.voice, args.max_tokens)
            passed = report(f"concurrency={n}", results)
            all_passed = all_passed and passed
        return 0 if all_passed else 1

    results = await run_once(args.url, args.concurrency, args.text, args.voice, args.max_tokens)
    passed = report(f"concurrency={args.concurrency}", results)
    return 0 if passed else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8080", help="FastAPI base URL")
    p.add_argument("--concurrency", type=int, default=50, help="Parallel requests")
    p.add_argument("--sweep", default=None, help="Comma-separated concurrency steps, e.g. 1,5,10,25,50,100")
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--voice", default=DEFAULT_VOICE)
    p.add_argument("--max-tokens", type=int, default=None)
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
