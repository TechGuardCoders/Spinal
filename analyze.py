#!/usr/bin/env python3
"""Spinal analyzer: reads a trace file, prints phase summaries (queue/prefill/
decode/total), correlates phases with engine concurrency, and runs drift
checks against baseline trace files if present.

Usage:
    python analyze.py traces/spinal_XXXX.json [--baseline traces/spinal_BASE.json]
"""
import glob
import json
import os
import statistics
import sys


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    return sorted_vals[min(len(sorted_vals) - 1, int(p * len(sorted_vals)))]


def phase_durations(traces, phase):
    out = []
    for t in traces:
        if t.get("failed"):
            continue
        ttfb, ttft, total = t.get("ttfbMs"), t.get("ttftMs"), t.get("totalMs")
        if None in (ttft, total):
            continue
        ttfb = ttfb if ttfb is not None else ttft
        if phase == "queue":
            out.append(min(ttfb, ttft))
        elif phase == "prefill":
            out.append(max(0, ttft - ttfb))
        elif phase == "decode":
            out.append(max(0, total - ttft))
        elif phase == "total":
            out.append(total)
    return sorted(out)


def summarize(traces):
    rows = []
    for phase in ("queue", "prefill", "decode", "total"):
        vals = phase_durations(traces, phase)
        rows.append({
            "phase": phase,
            "count": len(vals),
            "p50": pct(vals, 0.50),
            "p95": pct(vals, 0.95),
            "mean": round(statistics.mean(vals)) if vals else None,
        })
    return rows


def print_table(rows, title):
    print(f"\n=== {title} ===")
    print(f"{'phase':>8} {'count':>6} {'p50 ms':>8} {'p95 ms':>8} {'mean ms':>8}")
    for r in rows:
        print(f"{r['phase']:>8} {r['count']:>6} {str(r['p50']):>8} {str(r['p95']):>8} {str(r['mean']):>8}")


def correlation(traces):
    """Does decode time rise with engine concurrency at request time?"""
    pairs = []
    for t in traces:
        if t.get("failed") or not t.get("ttftMs") or not t.get("totalMs"):
            continue
        decode = t["totalMs"] - t["ttftMs"]
        ctok = t.get("completionTokens") or 0
        if ctok > 0:
            pairs.append((t.get("peakConcurrency", 0), decode / ctok * 1000))  # ms/token
    if len(pairs) < 4:
        return None
    by_c = {}
    for c, ms_per_tok in pairs:
        by_c.setdefault(c, []).append(ms_per_tok)
    return {c: round(statistics.mean(v), 1) for c, v in sorted(by_c.items())}


def drift_check(baseline_file, current_traces):
    """Compare total-ms distribution vs a baseline trace file."""
    if not baseline_file or not os.path.exists(baseline_file):
        return None
    with open(baseline_file) as f:
        base = json.load(f)
    base_totals = phase_durations(base["traces"], "total")
    cur_totals = phase_durations(current_traces, "total")
    if not base_totals or not cur_totals:
        return None
    b = statistics.mean(base_totals)
    c = statistics.mean(cur_totals)
    ratio = c / b if b else None
    return {"baseline_mean_ms": round(b), "current_mean_ms": round(c), "ratio": round(ratio, 3) if ratio else None}


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python analyze.py traces/<file>.json [--baseline traces/<base>.json]")
    path = sys.argv[1]
    baseline = None
    if "--baseline" in sys.argv:
        baseline = sys.argv[sys.argv.index("--baseline") + 1]

    with open(path) as f:
        doc = json.load(f)
    traces = doc["traces"]
    ok = [t for t in traces if not t.get("failed")]

    print(f"Spinal analysis: {path}")
    print(f"requests: {len(traces)} ({len(ok)} ok, {len(traces)-len(ok)} failed) "
          f"concurrency={doc.get('concurrency')} model={doc.get('model')}")

    print_table(summarize(traces), "PHASE TIMING (ms)")

    corr = correlation(traces)
    if corr:
        print("\n=== decode ms/token BY ENGINE CONCURRENCY ===")
        for c, ms in corr.items():
            print(f"  running={c}: {ms} ms/token")

    # workload-class breakdown
    by_cls = {}
    for t in ok:
        by_cls.setdefault(t.get("workloadClass", "?"), []).append(t)
    if by_cls:
        print("\n=== BY WORKLOAD CLASS (total ms) ===")
        for cls, ts in sorted(by_cls.items()):
            vals = sorted(t["totalMs"] for t in ts)
            print(f"  {cls:>7}: n={len(ts)} p50={pct(vals,0.5)} p95={pct(vals,0.95)}")

    d = drift_check(baseline, traces)
    if d:
        flag = "DRIFT DETECTED" if d["ratio"] and d["ratio"] > 1.25 else "within tolerance"
        print(f"\n=== DRIFT vs {os.path.basename(baseline)} ===")
        print(f"  baseline {d['baseline_mean_ms']}ms -> current {d['current_mean_ms']}ms "
              f"(x{d['ratio']}) = {flag}")


if __name__ == "__main__":
    main()
