# Spinal Benchmark - 2026-09-16

**Cluster:** 2x DGX Spark, GLM-5.3-Flash (EXL3 4-bit, TP=2), vLLM + DFlash2 speculation
**Method:** client-side phase tracing (TTFB/TTFT/total) + engine-state sampling (1s poll of /metrics gauges) during each request's window. No instrumentation inside vLLM.
**Raw data:** `traces/spinal_2026-09-16_225609.json` (c=2), `traces/spinal_2026-09-16_225647.json` (c=4)

## Phase timing (12 requests each run)

| Phase | c=2 p50 | c=2 p95 | c=4 p50 | c=4 p95 |
|---|---:|---:|---:|---:|
| queue (TTFB) | 27 ms | 34 ms | 27 ms | **1,035 ms** |
| prefill (TTFT-TTFB) | 480 ms | 514 ms | 685 ms | 746 ms |
| decode (total-TTFT) | 5,616 ms | 7,838 ms | 7,771 ms | 12,570 ms |
| **total** | **6,131 ms** | 8,333 ms | **8,521 ms** | 12,864 ms |

## Findings

**1. Queue time is invisible until it suddenly isn't.** At c=2, TTFB p50 was
27 ms - the engine accepts connections instantly. At c=4, p95 queue time
exploded to 1,035 ms (38x). This is the "non-determinism" the phase spine
exists to catch: total latency regressions get blamed on the model, when the
queue is where time actually went. Without splitting phases, a 1.5x total
regression is undiagnosable.

**2. Drift detection fired correctly.** c=4 run vs c=2 baseline: mean total
5,195 ms -> 7,669 ms = **x1.476, DRIFT DETECTED** (>1.25 tolerance). In
production this is the alert that pages someone; here it correctly flags the
expected cost of added concurrency.

**3. Decode dominates; prefill is small at these prompt sizes.** Prefill
(~450-570 ms) is under 10% of total. At 10k-token prompts this phase would
grow 20x; the spine makes that transition visible instead of inferred.

**4. Decode ms/token degrades with engine concurrency.** ~128-144 ms/token
at running=2 vs ~208-214 ms/token at running=3-4 - consistent with Batcher's
bandwidth-bound finding: batched streams share the same weight-read
bandwidth.

**5. Workload-class split validates the mixed workload:** short requests
p50 ~0.6-1.7s vs long ~6.5-10.3s. Real traffic is heterogeneous; benchmarks
that use uniform prompts hide the tail that short prompts never see.

## Caveats

- Phase boundaries are client-side approximations (queue = TTFB, prefill =
  TTFT-TTFB, decode = TTFT-end). Engine-internal splits (actual scheduler
  queue vs socket time) require in-engine instrumentation, which the
  no-touch-serving-stack rule forbids. The approximation is documented.
- Engine samples are 1s-resolution; a request can start and end between
  polls. Peak concurrency is a lower bound.
- The first "running=0.0" ms/token row is an artifact of a request whose
  window missed the sampler (single sample at 0). Trust the running=2+ rows.
