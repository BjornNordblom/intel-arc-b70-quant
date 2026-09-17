"""Smoke test for the swift-b70-mtp vLLM server: correctness, decode rate, MTP.

BASE defaults to the launchers' port; override for other hosts/ports:
    BASE=http://<host>:8080/v1 .venv/bin/python test_serve.py
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("MODEL", "swift38")


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def non_streaming():
    body = {"model": MODEL, "messages": [{"role": "user", "content": "In one sentence: what is 17*23?"}],
            "max_tokens": 64, "temperature": 0.0}
    t0 = time.time()
    out = json.loads(post("/chat/completions", body).read())
    dt = time.time() - t0
    msg = out["choices"][0]["message"]
    text = (msg.get("content") or msg.get("reasoning_content") or "")
    usage = out["usage"]
    print(f"[non-stream] {dt:.2f}s completion_tokens={usage['completion_tokens']} "
          f"({usage['completion_tokens']/dt:.1f} tok/s incl. prefill)")
    print(f"[non-stream] answer: {text.strip()[:200]!r}")
    return text


def streaming(max_tokens=256):
    body = {"model": MODEL, "messages": [{"role": "user", "content": "Write a haiku about Intel Arc GPUs."}],
            "max_tokens": max_tokens, "temperature": 0.7, "stream": True}
    t0 = time.time()
    first = None
    n = 0
    with post("/chat/completions", body) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                if first is None:
                    first = time.time()
                n += 1
    end = time.time()
    if first and n > 1:
        print(f"[stream] {n} chunks, TTFT {first-t0:.2f}s, post-first decode {n/(end-first):.1f} tok/s")
    else:
        print(f"[stream] {n} chunks, total {end-t0:.2f}s")
    return n


if __name__ == "__main__":
    text = non_streaming()
    n = streaming()
    if not text or n == 0:
        print("SMOKE FAIL: empty response")
        sys.exit(1)
    print("SMOKE PASS")
