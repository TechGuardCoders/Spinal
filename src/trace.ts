/**
 * Spinal — trace model for the inference serving path.
 *
 * A trace follows ONE request through the serving lifecycle:
 *   queue → prefill → decode (stream) → done
 *
 * On this cluster we cannot instrument inside vLLM (that would mean touching
 * the serving stack). So Spinal builds traces from OUTSIDE observations:
 * client-side timing of each request phase + the serving metrics deltas
 * (queue depth, KV cache) sampled during the request's life.
 *
 * This is exactly how black-box APM works and is honest about its blind spots
 * (we see total prefill+decode, not the engine's internal split).
 */

export type Phase = "queue" | "prefill" | "decode" | "total";

export interface Span {
  phase: Phase;
  /** ms since trace start */
  startMs: number;
  /** ms since trace start */
  endMs: number;
}

export interface RequestTrace {
  id: string;
  ts: number;
  /** prompt size in tokens (from usage) */
  promptTokens: number;
  /** completion size in tokens (from usage) */
  completionTokens: number;
  /** which workload class this request came from (short/medium/code/long) */
  workloadClass: string;
  spans: Span[];
  /** server-reported metrics sampled during the request (concurrency, kv) */
  samples: { ts: number; running: number; waiting: number; kvCache: number }[];
  /** engine-observed concurrency when this request was in flight (max) */
  peakConcurrency: number;
  /** true if the request failed or timed out */
  failed: boolean;
  error?: string;
}

/**
 * Build spans from client-side observations.
 *
 * Phase timing without engine internals:
 *  - queue:   submit → server accepted (we approximate with TTFB from the
 *             socket: request written → response headers). vLLM accepts fast;
 *             long TTFB under load ≈ queued.
 *  - prefill: TTFB → first content token (TTFT - TTFB). For vLLM, after
 *             acceptance the engine prefills the prompt before the first
 *             token. Approximation: prefill ≈ TTFT − TTFB.
 *  - decode:  first token → last token (stream duration).
 */
export function buildSpans(ttfbMs: number, ttftMs: number, totalMs: number): Span[] {
  return [
    { phase: "queue", startMs: 0, endMs: Math.min(ttfbMs, ttftMs) },
    { phase: "prefill", startMs: Math.min(ttfbMs, ttftMs), endMs: ttftMs },
    { phase: "decode", startMs: ttftMs, endMs: totalMs },
    { phase: "total", startMs: 0, endMs: totalMs },
  ];
}

/** Aggregate traces into phase-time summaries (ms). */
export interface PhaseSummary {
  phase: Phase;
  count: number;
  p50: number | null;
  p95: number | null;
  mean: number | null;
}

export function summarizePhases(traces: RequestTrace[]): PhaseSummary[] {
  const phases: Phase[] = ["queue", "prefill", "decode", "total"];
  return phases.map((phase) => {
    const durations = traces
      .filter((t) => !t.failed)
      .map((t) => {
        const span = t.spans.find((s) => s.phase === phase);
        return span ? span.endMs - span.startMs : null;
      })
      .filter((v): v is number => v !== null)
      .sort((a, b) => a - b);

    const pick = (p: number) =>
      durations.length ? Math.round(durations[Math.min(durations.length - 1, Math.floor(p * durations.length))]) : null;

    return {
      phase,
      count: durations.length,
      p50: pick(0.5),
      p95: pick(0.95),
      mean: durations.length ? Math.round(durations.reduce((a, b) => a + b, 0) / durations.length) : null,
    };
  });
}

/**
 * Drift detection: compare a recent window against a baseline window and
 * flag when a metric regresses beyond a tolerance. This is how "the serving
 * stack silently got slower" becomes a page instead of a shrug.
 */
export interface DriftCheck {
  metric: string;
  baseline: number;
  recent: number;
  /** recent/baseline ratio */
  ratio: number;
  /** true if ratio exceeds the direction-appropriate tolerance */
  drifted: boolean;
  direction: "up" | "down";
}

export function detectDrift(
  baseline: number[],
  recent: number[],
  tolerance = 1.25,
  direction: "up" | "down" = "up"
): DriftCheck | null {
  const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
  const b = mean(baseline);
  const r = mean(recent);
  if (b === null || r === null || b === 0) return null;
  const ratio = r / b;
  const drifted = direction === "up" ? ratio > tolerance : ratio < 1 / tolerance;
  return { metric: "", baseline: b, recent: r, ratio, drifted, direction };
}
