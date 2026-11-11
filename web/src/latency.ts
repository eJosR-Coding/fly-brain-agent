/** Latency map: how early each neuron fired first. Pure helpers; the scene just draws the levels. */
import type { Activity } from "./activity";

/** Brightness of the latest possible first spike: dim, but still on screen. */
export const LATENCY_FLOOR = 0.2;

/** First-spike time (ms, start of the bin) of every recorded row; NaN for a row that never spiked. */
export function firstSpikeMs(a: Activity): Float32Array {
  const out = new Float32Array(a.nRows);
  for (let i = 0; i < a.nRows; i++) {
    out[i] = NaN;
    for (let k = 0; k < a.nBins; k++) {
      if (a.counts[i * a.nBins + k]) {
        out[i] = k * a.binMs;
        break;
      }
    }
  }
  return out;
}

/** Brightness 0..1 for a first-spike time: 1 at 0 ms, falling linearly to LATENCY_FLOOR at the end of the run. */
export function latencyLevel(ms: number, durationMs: number): number {
  if (!Number.isFinite(ms) || !(durationMs > 0)) return 0;
  const f = Math.min(Math.max(ms / durationMs, 0), 1);
  return 1 - (1 - LATENCY_FLOOR) * f;
}

/** Write each row's latency level into `out` (same row order as the activity). */
export function fillLatency(a: Activity, out: Float32Array): void {
  if (out.length < a.nRows) throw new Error(`output buffer holds ${out.length} rows, the recording has ${a.nRows}`);
  const first = firstSpikeMs(a);
  for (let i = 0; i < a.nRows; i++) out[i] = latencyLevel(first[i], a.durationMs);
}

export function latencyLegend(durationMs: number): string {
  return `first spike · 0 ms → ${Math.round(durationMs)} ms`;
}

/** Tick labels under the gradient bar: both ends and the middle. */
export function latencyTicks(durationMs: number): { frac: number; label: string }[] {
  return [0, 0.5, 1].map((frac) => ({ frac, label: `${Math.round(frac * durationMs)} ms` }));
}

/** Grey levels (0-255) of the legend gradient, from the same scale as the neurons so the bar reads true. */
export function latencyGradientStops(steps = 4): { frac: number; grey: number }[] {
  return Array.from({ length: steps + 1 }, (_, i) => {
    const frac = i / steps;
    return { frac, grey: Math.round(255 * latencyLevel(frac, 1)) };
  });
}
