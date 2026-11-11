/** Difference map: what a silencing run lost and gained compared with its baseline. Pure; the scene draws the result. */
import type { Activity } from "./activity";

export const DIFF_KIND = { unchanged: 0, lost: 1, gained: 2, silenced: 3 } as const;
/** Sprite per neuron in the activity layer (the shader picks the texture by this code). Shape carries the meaning, never hue. */
export const SHAPE = { dot: 0, ring: 1, cross: 2 } as const;

/** Mean firing rate of one recorded row: spikes per trial per second. */
export function rateHz(a: Activity, i: number): number {
  let sum = 0;
  const base = i * a.nBins;
  for (let k = 0; k < a.nBins; k++) sum += a.counts[base + k];
  return sum / (Math.max(1, a.trials) * (a.durationMs / 1000));
}

export interface Diff {
  /** union of both recordings' rows (plus the silenced neurons), ascending */
  rows: Uint32Array;
  /** silenced minus baseline rate, Hz */
  delta: Float32Array;
  kind: Uint8Array;
  lost: number;
  gained: number;
  /** largest |delta| among lost/gained neurons (0 when nothing changed) */
  maxAbs: number;
}

/**
 * Per neuron: rate in the silenced run minus rate in the baseline. |delta| below `minHz` counts as unchanged.
 * The silenced neurons are marked on their own and kept out of the lost/gained counts.
 */
export function computeDiff(base: Activity, sil: Activity, silencedRows: ArrayLike<number>, minHz = 0.5): Diff {
  if (base.nBins !== sil.nBins || base.binMs !== sil.binMs || base.trials !== sil.trials) {
    throw new Error("the two recordings have different time bases (bins, bin width or trials)");
  }
  const silenced = new Set<number>(Array.from(silencedRows));
  const all = new Set<number>([...base.rows, ...sil.rows, ...silenced]);
  const rows = Uint32Array.from([...all].sort((x, y) => x - y));
  const baseAt = new Map<number, number>();
  base.rows.forEach((r, i) => baseAt.set(r, i));
  const silAt = new Map<number, number>();
  sil.rows.forEach((r, i) => silAt.set(r, i));
  const delta = new Float32Array(rows.length);
  const kind = new Uint8Array(rows.length);
  let lost = 0;
  let gained = 0;
  let maxAbs = 0;
  rows.forEach((row, i) => {
    const b = baseAt.has(row) ? rateHz(base, baseAt.get(row)!) : 0;
    const s = silAt.has(row) ? rateHz(sil, silAt.get(row)!) : 0;
    delta[i] = s - b;
    if (silenced.has(row)) {
      kind[i] = DIFF_KIND.silenced;
      return;
    }
    const mag = Math.abs(delta[i]);
    if (mag === 0 || mag < minHz) return;
    kind[i] = delta[i] < 0 ? DIFF_KIND.lost : DIFF_KIND.gained;
    if (delta[i] < 0) lost++;
    else gained++;
    if (mag > maxAbs) maxAbs = mag;
  });
  return { rows, delta, kind, lost, gained, maxAbs };
}

const DIFF_FLOOR = 0.35; // smallest visible change is still readable
const SILENCED_GREY = 0.55; // the silenced marks are a fixed mid grey, distinct from any change

/** Brightness (0..1) per row: unchanged hidden, changes scale with the magnitude (sqrt keeps small ones visible). */
export function diffBrightness(d: Diff, out: Float32Array): void {
  if (out.length < d.rows.length) throw new Error(`output buffer holds ${out.length} rows, the diff has ${d.rows.length}`);
  for (let i = 0; i < d.rows.length; i++) {
    if (d.kind[i] === DIFF_KIND.silenced) out[i] = SILENCED_GREY;
    else if (d.kind[i] === DIFF_KIND.unchanged) out[i] = 0;
    else out[i] = DIFF_FLOOR + (1 - DIFF_FLOOR) * Math.sqrt(Math.abs(d.delta[i]) / d.maxAbs);
  }
}

/** Sprite code per row: lost = ring, gained = filled dot, silenced = cross. */
export function diffShapes(d: Diff, out: Float32Array): void {
  if (out.length < d.rows.length) throw new Error(`output buffer holds ${out.length} rows, the diff has ${d.rows.length}`);
  for (let i = 0; i < d.rows.length; i++) {
    out[i] = d.kind[i] === DIFF_KIND.lost ? SHAPE.ring : d.kind[i] === DIFF_KIND.silenced ? SHAPE.cross : SHAPE.dot;
  }
}

/** Mean of a per-bin trace (the readout group's Hz). */
export function meanHz(hz: readonly number[]): number {
  return hz.length ? hz.reduce((a, b) => a + b, 0) / hz.length : 0;
}

export function diffLegend(lost: number, gained: number, readout: { label: string; before: number; after: number } | null): string {
  const n = (v: number) => v.toLocaleString("en-US");
  const base = `diff · ${n(lost)} lost · ${n(gained)} gained`;
  return readout ? `${base} · ${readout.label} ${Math.round(readout.before)} → ${Math.round(readout.after)} Hz` : base;
}
