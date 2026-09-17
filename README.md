# Spinal

**Observability spine for non-determinism: phase traces + drift detection for a self-hosted inference cluster.**

Traces each request through queue -> prefill -> decode, correlates phases
with engine state (concurrency, KV cache), and alerts when serving drifts
from baseline. Because you can't page what you can't see.

![CI](https://github.com/TechGuardCoders/Spinal/actions/workflows/ci.yml/badge.svg)

## Why

LLM serving latency is non-deterministic: the same prompt can take 2s or 12s
depending on what else the engine is doing. A total-latency chart tells you
THAT it got slower, not WHERE the time went. Spinal splits every request into
phases - so a 1.5x regression becomes "queue time exploded 38x" (capacity
problem) instead of "the model got slower" (wrong culprit, wrong fix).

## The trace model

```
request
  ├─ queue    submit → server accepted (TTFB). Long here = engine saturated.
  ├─ prefill  accepted → first token (TTFT − TTFB). Scales with prompt size.
  ├─ decode   first token → last token. Scales with output length × concurrency.
  └─ total
```

Phases are measured CLIENT-SIDE and correlated with engine gauges sampled
during each request's window (`num_requests_running`, `num_requests_waiting`,
`kv_cache_usage_perc`). No instrumentation inside vLLM - the serving stack is
never touched. The approximation is documented: queue = TTFB, prefill =
TTFT−TTFB, decode = TTFT→end.

## What it found (2026-09-16, real traces)

| Phase | c=2 p50 | c=4 p50 |
|---|---:|---:|
| queue | 27 ms | 27 ms (p95: **1,035 ms**, 38×) |
| prefill | 480 ms | 685 ms |
| decode | 5,616 ms | 7,771 ms |
| total | 6,131 ms | 8,521 ms |

Drift detection vs baseline: **×1.476 = DRIFT DETECTED** at c=4. Decode
ms/token degrades ~1.6× from running=2 → running=4, consistent with the
bandwidth-bound finding in [Batcher](https://github.com/TechGuardCoders/Batcher).
Full analysis: `traces/2026-09-16-spinal-analysis.md`.

## Run it

```bash
# trace: fire real requests + sample engine state
VLLM_API_KEY=<key> python tracer.py --requests 12 --concurrency 2 --max-tokens 120

# analyze: phase table, concurrency correlation, workload split, drift
python analyze.py traces/spinal_<ts>.json [--baseline traces/spinal_<base>.json]
```

Drift tolerance is 1.25× by default (upward) — the threshold that turned
"c=4 is slower" into a flagged, quantified alert.

## How it serves the stack

- **Queue-time visibility** — the phase that's invisible in total-latency
  charts and is the first casualty of saturation
- **Drift alerts** — compare any run against a baseline; ratio > 1.25 flags
- **Concurrency correlation** — decode cost per token by engine load, the
  input to autoscaler thresholds
- **Zero serving-stack footprint** — everything is client timing + /metrics
  polls, per the no-touch rule

## Tests

```bash
python tests/test_logic.py   # workload mixer, gauge parser (offline)
```
