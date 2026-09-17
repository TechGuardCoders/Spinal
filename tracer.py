#!/usr/bin/env python3
"""Spinal tracer: drives REAL requests through the serving path and records
per-request phase traces (queue/prefill/decode) + engine samples.

Works with the traffic driver: while requests are in flight, a sampler polls
vLLM /metrics for concurrency + KV cache. Client timing (TTFB, TTFT, total)
is captured per request. Together: request-level traces correlated with
engine state — without instrumenting the serving stack.

Usage:
    VLLM_API_KEY=<key> python tracer.py --requests 12 --concurrency 2
    # then: python analyze.py traces/<file>.json
"""
import argparse
import concurrent.futures
import json
import os
import random
import sys
import threading
import time
import urllib.request
import urllib.error
import uuid

sys.path.insert(0, os.path.dirname(__file__))
from workload import PROMPT_CLASSES, pick_prompt  # noqa: E402
from sampler import MetricsSampler  # noqa: E402


def post_chat_streaming(base, key, model, prompt, max_tokens, timeout=240):
    """Streaming chat completion, timed at: TTFB, TTFT, total."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": True,
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    trace = {
        "id": uuid.uuid4().hex[:12],
        "ts": time.time(),
        "workloadClass": None,
        "promptTokens": 0,
        "completionTokens": 0,
        "ttfbMs": None,
        "ttftMs": None,
        "totalMs": None,
        "failed": False,
        "error": None,
    }
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            # TTFB: response headers received (server accepted the request)
            trace["ttfbMs"] = round((time.perf_counter() - t0) * 1000)
            first_token = True
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line.endswith("[DONE]"):
                    continue
                try:
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0].get("delta", {})
                except Exception:
                    continue
                if first_token and (delta.get("content") or delta.get("role")):
                    trace["ttftMs"] = round((time.perf_counter() - t0) * 1000)
                    first_token = False
                if delta.get("content"):
                    trace["completionTokens"] += 1
            trace["totalMs"] = round((time.perf_counter() - t0) * 1000)
        trace["ok"] = True
    except Exception as e:  # noqa: BLE001
        trace["failed"] = True
        trace["error"] = f"{type(e).__name__}: {e}"
        trace["totalMs"] = round((time.perf_counter() - t0) * 1000)
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sample-interval", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = os.environ.get("VLLM_BASE_URL", "http://192.168.0.182:8000")
    key = os.environ.get("VLLM_API_KEY")
    model = os.environ.get("VLLM_MODEL", "glm-5.3-flash")
    if not key:
        sys.exit("set VLLM_API_KEY; never committed")

    sampler = MetricsSampler(base, interval=args.sample_interval)
    sampler.start()

    print(f"Spinal tracer: {args.requests} requests, concurrency {args.concurrency}")
    rng_seed = 42
    traces = []
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = []
        for i in range(args.requests):
            cls, prompt = pick_prompt(random.Random(rng_seed + i))
            futures.append((cls, ex.submit(post_chat_streaming, base, key, model, prompt, args.max_tokens)))
        for cls, fut in futures:
            tr = fut.result()
            tr["workloadClass"] = cls
            traces.append(tr)

    wall = time.perf_counter() - t0
    sampler.stop()

    # attach engine samples within each request's window
    samples = sampler.samples
    for tr in traces:
        tr["samples"] = [
            s for s in samples
            if tr["ts"] <= s["ts"] <= tr["ts"] + (tr["totalMs"] or 0) / 1000
        ]
        tr["peakConcurrency"] = max((s["running"] for s in tr["samples"]), default=0)

    ok = [t for t in traces if not t["failed"]]
    doc = {
        "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
        "endpoint": base,
        "model": model,
        "concurrency": args.concurrency,
        "wall_s": round(wall, 2),
        "traces": traces,
        "engine_samples": samples,
    }
    out = args.out or os.path.join("traces", f"spinal_{doc['timestamp']}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"ok: {len(ok)}/{len(traces)}  wall: {wall:.1f}s  -> {out}")


if __name__ == "__main__":
    main()
