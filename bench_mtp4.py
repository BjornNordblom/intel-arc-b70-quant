"""Benchmark the MTP server using the cookbook method: p512/g128, median n=5.

tokSOut = client post-first-token rate: tokens after first / (last_chunk - first_chunk).

BASE defaults to the launchers' port; override for other hosts/ports:
    BASE=http://<host>:8080/v1 .venv/bin/python bench_mtp4.py
"""
import json
import os
import statistics
import time
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("MODEL", "swift38")
PROMPT_WORDS = 380  # ~512 tokens
GEN = 128
N = 5

PROMPT = ("Explain the history of computing in detail. " * 1) + " ".join(
    f"Topic {i} covers an important aspect of computer architecture and software systems." for i in range(PROMPT_WORDS // 11)
)


def run(i):
    body = {"model": MODEL, "prompt": PROMPT, "max_tokens": GEN, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    last = None
    tokens = None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
            choices = chunk.get("choices") or []
            if choices and (choices[0].get("text") or choices[0].get("delta", {}).get("content")):
                now = time.time()
                if first is None:
                    first = now
                last = now
    rate = (tokens - 1) / (last - first) if first and last and tokens and last > first else 0.0
    ttft = (first - t0) if first else 0.0
    print(f"run {i}: tokens={tokens} ttft={ttft:.2f}s decode={rate:.1f} tok/s", flush=True)
    return rate


rates = [run(i) for i in range(1, N + 1)]
print(f"\nmedian post-first decode: {statistics.median(rates):.1f} tok/s (n={N})")
print(f"range: {min(rates):.1f}-{max(rates):.1f}")
