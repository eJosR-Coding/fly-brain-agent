/** Pure data shaping for the charts panel: raster rows, readout trace, scales, ticks and tooltip text. */
import type { Activity } from "./activity";

/** Maps `d0..d1` onto `r0..r1`, clamped to the domain. An empty domain maps everything to `r0`. */
export function linearScale(d0: number, d1: number, r0: number, r1: number): (v: number) => number {
  if (d1 === d0) return () => r0;
  return (v) => r0 + ((Math.min(Math.max(v, Math.min(d0, d1)), Math.max(d0, d1)) - d0) / (d1 - d0)) * (r1 - r0);
}

/** A 1-2-5 axis from 0: the top is rounded up to a whole step, ticks include 0 and the top. */
export function niceTicks(max: number, target = 4): { max: number; ticks: number[] } {
  const top = Number.isFinite(max) && max > 0 ? max : 1;
  const raw = top / Math.max(1, target);
  const pow = 10 ** Math.floor(Math.log10(raw));
  const step = ([1, 2, 5, 10].find((m) => m * pow >= raw) ?? 10) * pow;
  const n = Math.ceil(top / step - 1e-9);
  const round = (v: number) => Math.round(v * 1e9) / 1e9;
  return { max: round(n * step), ticks: Array.from({ length: n + 1 }, (_, i) => round(i * step)) };
}

// --- raster --------------------------------------------------------------------------------

export interface RasterRow { row: number; label: string; emphasized: boolean; total: number; spikes: { bin: number; count: number }[] }
export interface RasterGroup { label: string; start: number; end: number; emphasized: boolean }
export interface RasterModel { rows: RasterRow[]; groups: RasterGroup[]; durationMs: number; binMs: number; nBins: number }

const MAX_EMPHASIZED = 3;

/**
 * The raster's rows: the `top` most active neurons by total spikes (ties by row), plus up to a few rows of the
 * emphasized cell type (the readout) even when quiet. Rows are grouped by cell type, the group with the busiest
 * neuron first, so a label can mark a whole group instead of every row.
 */
export function rasterModel(
  a: Activity, typeOf: (row: number) => string | null, opts: { top?: number; emphasize?: { rows: readonly number[]; label: string } | null } = {},
): RasterModel {
  const top = opts.top ?? 30;
  const totals = new Float64Array(a.nRows);
  for (let i = 0; i < a.nRows; i++) for (let k = 0; k < a.nBins; k++) totals[i] += a.counts[i * a.nBins + k];
  const byActivity = Array.from({ length: a.nRows }, (_, i) => i)
    .filter((i) => totals[i] > 0)
    .sort((x, y) => totals[y] - totals[x] || a.rows[x] - a.rows[y]);
  const chosen = new Set(byActivity.slice(0, Math.max(0, top)));
  const marked = new Set(opts.emphasize?.rows ?? []);
  const label = (r: number) => (marked.has(r) ? opts.emphasize!.label : (typeOf(r) ?? "unknown"));
  const recorded = new Map<number, number>(Array.from(a.rows, (r, i) => [r, i]));
  const spikesOf = (i: number) => {
    const spikes: { bin: number; count: number }[] = [];
    for (let k = 0; k < a.nBins; k++) {
      const c = a.counts[i * a.nBins + k];
      if (c) spikes.push({ bin: k, count: c });
    }
    return spikes;
  };
  type Entry = { row: number; total: number; spikes: { bin: number; count: number }[] };
  const entry = (i: number): Entry => ({ row: a.rows[i], total: totals[i], spikes: spikesOf(i) });
  const entries: Entry[] = byActivity.filter((i) => chosen.has(i)).map(entry);
  if (opts.emphasize) {
    // the readout stays visible even when it is quiet: a neuron that never spiked is a flat row, not missing data
    const extras = opts.emphasize.rows
      .filter((r) => !entries.some((e) => e.row === r))
      .sort((x, y) => (recorded.has(y) ? totals[recorded.get(y)!] : -1) - (recorded.has(x) ? totals[recorded.get(x)!] : -1) || x - y)
      .slice(0, MAX_EMPHASIZED);
    for (const r of extras) entries.push(recorded.has(r) ? entry(recorded.get(r)!) : { row: r, total: 0, spikes: [] });
  }
  entries.sort((x, y) => y.total - x.total || x.row - y.row);
  const groupsByLabel = new Map<string, Entry[]>(); // insertion order = busiest first
  for (const e of entries) groupsByLabel.set(label(e.row), [...(groupsByLabel.get(label(e.row)) ?? []), e]);
  const rows: RasterRow[] = [];
  const groups: RasterGroup[] = [];
  for (const [name, members] of groupsByLabel) {
    const emphasized = opts.emphasize !== undefined && opts.emphasize !== null && members.some((e) => marked.has(e.row));
    groups.push({ label: name, start: rows.length, end: rows.length + members.length, emphasized });
    for (const e of members) rows.push({ row: e.row, label: name, emphasized: marked.has(e.row), total: e.total, spikes: e.spikes });
  }
  return { rows, groups, durationMs: a.nBins * a.binMs, binMs: a.binMs, nBins: a.nBins };
}

/** y (px) of each group's label, centered on its rows; a label that would sit closer than `minGapPx` to the previous one is dropped (emphasized ones never are). */
export function groupLabelLayout(groups: RasterGroup[], rowPx: number, minGapPx: number): { label: string; y: number; emphasized: boolean }[] {
  const out: { label: string; y: number; emphasized: boolean }[] = [];
  let lastY = -Infinity;
  for (const g of groups) {
    const y = ((g.start + g.end) / 2) * rowPx;
    if (!g.emphasized && y - lastY < minGapPx) continue;
    out.push({ label: g.label, y, emphasized: g.emphasized });
    lastY = y;
  }
  return out;
}

export function rasterTooltip(row: Pick<RasterRow, "label" | "row" | "spikes">, bin: number, binMs: number): string {
  const count = row.spikes.find((s) => s.bin === bin)?.count ?? 0;
  return `${row.label} · row ${row.row} · t = ${Math.round(bin * binMs)}–${Math.round((bin + 1) * binMs)} ms · ${count} spike${count === 1 ? "" : "s"}`;
}

// --- readout trace -------------------------------------------------------------------------

export interface TraceSeries { label: string; dashed: boolean; points: { t: number; hz: number }[] }
export interface TraceModel { series: TraceSeries[]; axis: { max: number; ticks: number[] }; durationMs: number; binMs: number }

/** One line: the group's mean rate per bin, drawn at the bin centers. */
export function readoutSeries(label: string, hz: readonly number[], binMs: number, dashed: boolean): TraceSeries {
  return { label, dashed, points: hz.map((v, i) => ({ t: (i + 0.5) * binMs, hz: v })) };
}

/** All series share ONE y axis (no dual axes), so baseline and silenced are directly comparable. */
export function traceModel(series: TraceSeries[], binMs: number, durationMs: number): TraceModel {
  const max = Math.max(0, ...series.flatMap((s) => s.points.map((p) => p.hz)));
  return { series, axis: niceTicks(max, 4), durationMs, binMs };
}

export function tracePath(s: TraceSeries, x: (t: number) => number, y: (hz: number) => number): string {
  return s.points.map((p, i) => `${i ? "L" : "M"}${x(p.t)},${y(p.hz)}`).join(" ");
}

export function traceTooltip(m: TraceModel, bin: number): string {
  const time = `t = ${Math.round(bin * m.binMs)}–${Math.round((bin + 1) * m.binMs)} ms`;
  return [time, ...m.series.map((s) => `${s.label} ${(s.points[bin]?.hz ?? 0).toFixed(1)} Hz`)].join(" · ");
}

/** Cell type per atlas row from `GET /simulations/{id}/rows`; malformed entries are skipped. */
export function parseRowTypes(json: unknown): Map<number, string | null> {
  const out = new Map<number, string | null>();
  if (!Array.isArray(json)) return out;
  for (const e of json) {
    if (typeof e !== "object" || e === null) continue;
    const r = e as Record<string, unknown>;
    if (typeof r.row !== "number" || !Number.isInteger(r.row)) continue;
    out.set(r.row, typeof r.cell_type === "string" && r.cell_type ? r.cell_type : null);
  }
  return out;
}

/** Centered moving average over `window` bins (edges average what is available); same length as the input. */
export function smooth(values: readonly number[], window: number): number[] {
  const half = Math.floor(Math.max(1, window) / 2);
  return values.map((_, i) => {
    const from = Math.max(0, i - half);
    const to = Math.min(values.length - 1, i + half);
    let sum = 0;
    for (let k = from; k <= to; k++) sum += values[k];
    return sum / (to - from + 1);
  });
}

/** Odd number of bins covering about `ms` (default 25 ms): the readout trace is a few spikes per bin, so it is averaged to be readable. */
export function smoothWindow(binMs: number, ms = 25): number {
  const n = Math.max(1, Math.round(ms / binMs));
  return n % 2 ? n : n + 1;
}
