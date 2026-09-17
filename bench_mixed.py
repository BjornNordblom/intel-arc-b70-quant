#!/usr/bin/env python3
"""Concurrency harness for the mixed spec-decode / prefill GDN bug.

Scenarios
  s1  solo        : one short prompt, measures TTFT + post-first-token decode rate
  s2  mixed       : request A decodes with MTP draft tokens while request B starts
                    a long, unique prefill (the exact crash trigger); repeated N times
  s3  storm       : 12 concurrent requests (long + short, staggered)

Pass/fail is decided by: HTTP/stream errors, engine death (container no longer
running), or the abort strings in the server log.

Only stdlib. Mirrors the TTFT/decode definitions used by bench_mtp4.py:
  TTFT  = time from request start to first non-empty content chunk
  rate  = (completion_tokens - 1) / (last_chunk - first_chunk)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ABORT_PATTERNS = (
    "mutually exclusive",
    "EngineDeadError",
    "EngineCore encountered a fatal error",
    "Overflow when unpacking long long",
    "RuntimeError",
)

SHORT_TOPICS = [
    "computer architecture",
    "operating system schedulers",
    "compiler optimisation passes",
    "cache coherence protocols",
    "instruction pipelining",
]


def make_short_prompt(rng: random.Random, words: int = 380) -> str:
    parts = ["Explain the history of computing in detail."]
    for i in range(words // 11):
        parts.append(
            f"Topic {rng.randrange(10**9)} covers an important aspect of "
            f"{rng.choice(SHORT_TOPICS)} and software systems."
        )
    return " ".join(parts)


def make_long_prompt(rng: random.Random, words: int) -> str:
    """Unique by construction so prefix caching cannot short-circuit the prefill."""
    parts = []
    for i in range(words // 10):
        parts.append(
            f"Record {rng.randrange(10**12)} entry {i} documents measurement "
            f"{rng.randrange(10**6)} of subsystem {rng.randrange(10**6)} "
            f"with notes on "
            f"{rng.choice(SHORT_TOPICS)}."
        )
    return " ".join(parts)


class Result:
    __slots__ = ("label", "ttft", "rate", "tokens", "error", "http_status")

    def __init__(self, label: str):
        self.label = label
        self.ttft: float | None = None
        self.rate: float | None = None
        self.tokens: int | None = None
        self.error: str | None = None
        self.http_status: int | None = None

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "ttft_s": None if self.ttft is None else round(self.ttft, 3),
            "decode_tok_s": None if self.rate is None else round(self.rate, 2),
            "completion_tokens": self.tokens,
            "http_status": self.http_status,
            "error": self.error,
        }


def request(base: str, model: str, prompt: str, max_tokens: int, label: str,
            timeout: float = 900.0) -> Result:
    res = Result(label)
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base + "/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    first = last = None
    tokens = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            res.http_status = resp.status
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
    except urllib.error.HTTPError as exc:
        res.error = f"HTTP {exc.code}: {exc.read()[:200]!r}"
        res.http_status = exc.code
        return res
    except Exception as exc:  # noqa: BLE001
        res.error = f"{type(exc).__name__}: {exc}"
        return res

    if first is not None and res.ttft is None:
        res.ttft = first - t0
    res.tokens = tokens
    if first and last and tokens and tokens > 1 and last > first:
        res.rate = (tokens - 1) / (last - first)
    elif tokens is not None and tokens <= 1:
        res.rate = 0.0
    return res


def run_parallel(fns) -> list[Result]:
    results: list[Result | None] = [None] * len(fns)

    def worker(i, fn):
        results[i] = fn()

    threads = [threading.Thread(target=worker, args=(i, fn)) for i, fn in enumerate(fns)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [r for r in results if r is not None]


def container_running(name: str) -> bool:
    try:
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip() == "true"
    except Exception:  # noqa: BLE001
        return False


def scrape_metrics(base: str) -> dict:
    wanted = (
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:num_requests_running",
        "vllm:num_preemptions_total",
        "vllm:request_success_total",
    )
    out: dict[str, float] = {}
    try:
        with urllib.request.urlopen(base.replace("/v1", "") + "/metrics", timeout=30) as resp:
            for line in resp.read().decode().splitlines():
                for key in wanted:
                    if line.startswith(key):
                        try:
                            out[key] = float(line.rsplit(" ", 1)[1])
                        except (IndexError, ValueError):
                            pass
    except Exception as exc:  # noqa: BLE001
        out["metrics_error"] = str(exc)  # type: ignore[assignment]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="swift38")
    ap.add_argument("--container", default="swift-b70-mtp-split")
    ap.add_argument("--outdir", default="artifacts/bench-mixed")
    ap.add_argument("--scenarios", default="s1,s2,s3")
    ap.add_argument("--stagger", type=float, default=1.5)
    ap.add_argument("--s2-repeat", type=int, default=3)
    ap.add_argument("--s2-long-words", type=int, default=8000)  # ~10-12k tokens
    ap.add_argument("--s3-requests", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed)
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    report: dict = {"config": vars(args), "scenarios": {}, "started": time.time()}

    def fail_fast() -> str | None:
        if not container_running(args.container):
            return f"container {args.container} is not running"
        return None

    # ---------------- s1 solo ----------------
    if "s1" in scenarios:
        print("== s1 solo ==", flush=True)
        r = request(args.base, args.model, make_short_prompt(rng), 128, "s1-solo")
        print("  ", r.as_dict(), flush=True)
        report["scenarios"]["s1"] = {"requests": [r.as_dict()], "metrics": scrape_metrics(args.base)}

    # ---------------- s2 mixed ----------------
    if "s2" in scenarios:
        print("== s2 mixed (staggered prefill while decoding with draft tokens) ==", flush=True)
        runs = []
        for i in range(args.s2_repeat):
            dead = fail_fast()
            if dead:
                report["scenarios"]["s2"] = {"requests": runs, "aborted": dead}
                print(f"  ABORT before run {i + 1}: {dead}", flush=True)
                break
            long_prompt = make_long_prompt(rng, args.s2_long_words)
            short_prompt = make_short_prompt(rng)
            t0 = time.time()
            a_holder: list[Result] = []
            b_holder: list[Result] = []

            def req_a():
                a_holder.append(request(args.base, args.model, short_prompt, 512, f"s2-{i}-decode"))

            def req_b():
                time.sleep(args.stagger)
                b_holder.append(request(args.base, args.model, long_prompt, 64, f"s2-{i}-prefill"))

            ta = threading.Thread(target=req_a)
            tb = threading.Thread(target=req_b)
            ta.start()
            tb.start()
            ta.join()
            tb.join()
            pair = {
                "run": i,
                "wall_s": round(time.time() - t0, 2),
                "decode": a_holder[0].as_dict() if a_holder else None,
                "prefill": b_holder[0].as_dict() if b_holder else None,
            }
            runs.append(pair)
            print(f"   s2 run {i}: {json.dumps(pair)}", flush=True)
            dead = fail_fast()
            if dead:
                pair["aborted_after"] = dead
                print(f"  ABORT after run {i + 1}: {dead}", flush=True)
                break
        report["scenarios"]["s2"] = {"requests": runs, "metrics": scrape_metrics(args.base)}

    # ---------------- s3 storm ----------------
    if "s3" in scenarios:
        print("== s3 storm ==", flush=True)
        dead = fail_fast()
        if dead:
            report["scenarios"]["s3"] = {"aborted": dead}
        else:
            long_n = max(2, args.s3_requests // 3)
            fns = []
            for i in range(args.s3_requests):
                if i < long_n:
                    prompt = make_long_prompt(rng, args.s2_long_words)
                    max_tokens = 64
                    label = f"s3-{i}-long"
                else:
                    prompt = make_short_prompt(rng)
                    max_tokens = 256
                    label = f"s3-{i}-short"
                delay = 0.2 * i

                def fn(p=prompt, m=max_tokens, lb=label, d=delay):
                    time.sleep(d)
                    return request(args.base, args.model, p, m, lb)

                fns.append(fn)
            t0 = time.time()
            results = run_parallel(fns)
            wall = time.time() - t0
            report["scenarios"]["s3"] = {
                "wall_s": round(wall, 2),
                "requests": [r.as_dict() for r in results],
                "metrics": scrape_metrics(args.base),
            }
            print(f"   s3 wall={wall:.1f}s results={json.dumps([r.as_dict() for r in results])}", flush=True)

    # ---------------- verdict ----------------
    errors = []
    ttfts = []
    for sc, blob in report["scenarios"].items():
        for r in blob.get("requests", []):
            entries = [r] if "error" in r else [v for v in (r.get("decode"), r.get("prefill")) if v]
            for e in entries:
                if e.get("error"):
                    errors.append(f"{sc}: {e['label']}: {e['error']}")
                if e.get("ttft_s") is not None:
                    ttfts.append(e["ttft_s"])
        if blob.get("aborted"):
            errors.append(f"{sc}: {blob['aborted']}")

    alive = container_running(args.container)
    report["verdict"] = {
        "engine_alive_at_end": alive,
        "errors": errors,
        "ttft_p50_s": round(statistics.median(ttfts), 3) if ttfts else None,
        "ttft_max_s": round(max(ttfts), 3) if ttfts else None,
        "pass": alive and not errors,
    }
    report["finished"] = time.time()

    with open(os.path.join(args.outdir, "bench.json"), "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    v = report["verdict"]
    print("\n== verdict ==", flush=True)
    print(f"  engine_alive={v['engine_alive_at_end']}  errors={len(errors)}  "
          f"ttft_p50={v['ttft_p50_s']}s  ttft_max={v['ttft_max_s']}s  pass={v['pass']}", flush=True)
    for e in errors:
        print(f"  ERROR {e}", flush=True)

    # server-log abort check
    try:
        logs = subprocess.run(["docker", "logs", args.container], capture_output=True,
                              text=True, timeout=60).stdout + subprocess.run(
            ["docker", "logs", args.container], capture_output=True, text=True, timeout=60).stderr
        with open(os.path.join(args.outdir, "server.log"), "w") as fh:
            fh.write(logs)
        hits = [p for p in ABORT_PATTERNS if p in logs]
        if hits:
            print(f"  server log abort patterns: {hits}", flush=True)
            report["verdict"]["server_log_abort_patterns"] = hits
            report["verdict"]["pass"] = False
            with open(os.path.join(args.outdir, "bench.json"), "w") as fh:
                json.dump(report, fh, indent=2, sort_keys=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  (log capture failed: {exc})", flush=True)

    return 0 if report["verdict"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
