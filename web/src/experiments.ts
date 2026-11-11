/** Pure data shaping for the experiment charts (dose-response curve, silencing screen) and their progress. */
import { linearScale, niceTicks } from "./charts";

export interface DosePoint { rate: number; mean: number; low: number | null; high: number | null }
export interface DoseChart {
  kind: "dose_response"; id: string; readoutLabel: string; stimulusLabel: string; trials: number; durationMs: number;
  ec50: number | null; points: DosePoint[];
}
export interface ScreenRow {
  label: string; row: number; delta: number; low: number | null; high: number | null; p: number | null;
  significant: boolean; simulationId: string;
}
export interface ScreenChart {
  kind: "screen"; id: string; readoutLabel: string; stimulusLabel: string; trials: number; durationMs: number;
  baselineId: string; rows: ScreenRow[];
}
export type ExperimentChart = DoseChart | ScreenChart;

const num = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const numOrNull = (v: unknown): number | null => (num(v) ? v : null);

/** Check a `chart` event payload. Returns the typed chart, or why it is invalid. */
export function parseChart(d: unknown): ExperimentChart | string {
  if (typeof d !== "object" || d === null) return "payload is not an object";
  const o = d as Record<string, unknown>;
  const common = {
    id: typeof o.id === "string" ? o.id : "",
    readoutLabel: typeof o.readout_label === "string" && o.readout_label ? o.readout_label : "readout",
    stimulusLabel: typeof o.stimulus_label === "string" && o.stimulus_label ? o.stimulus_label : "stimulus",
    trials: num(o.trials) ? o.trials : 1,
    durationMs: num(o.duration_ms) ? o.duration_ms : 0,
  };
  if (o.kind === "dose_response") {
    if (!Array.isArray(o.points) || o.points.length === 0) return "points is not a non-empty array";
    const points: DosePoint[] = [];
    for (const p of o.points as Record<string, unknown>[]) {
      if (typeof p !== "object" || p === null || !num(p.rate_hz) || p.rate_hz <= 0 || !num(p.mean_hz)) return "bad point";
      points.push({ rate: p.rate_hz, mean: p.mean_hz, low: numOrNull(p.ci_low), high: numOrNull(p.ci_high) });
    }
    points.sort((a, b) => a.rate - b.rate);
    return { kind: "dose_response", ...common, ec50: numOrNull(o.ec50_hz), points };
  }
  if (o.kind === "screen") {
    if (!Array.isArray(o.rows)) return "rows is not an array";
    const rows: ScreenRow[] = [];
    for (const r of o.rows as Record<string, unknown>[]) {
      if (typeof r !== "object" || r === null || typeof r.label !== "string" || !num(r.row) || !num(r.delta_hz) || typeof r.simulation_id !== "string") {
        return "bad row";
      }
      rows.push({
        label: r.label, row: r.row, delta: r.delta_hz, low: numOrNull(r.ci_low), high: numOrNull(r.ci_high), p: numOrNull(r.p),
        significant: r.significant === true, simulationId: r.simulation_id,
      });
    }
    return { kind: "screen", ...common, baselineId: typeof o.baseline_simulation_id === "string" ? o.baseline_simulation_id : "", rows };
  }
  return "unknown chart kind";
}

/** Log scale `d0..d1` onto `r0..r1` (clamped); a single-point domain maps to `r0`. */
export function logScale(d0: number, d1: number, r0: number, r1: number): (v: number) => number {
  if (!(d1 > d0) || d0 <= 0) return () => r0;
  const a = Math.log(d0);
  const span = Math.log(d1) - a;
  return (v) => r0 + ((Math.log(Math.min(Math.max(v, d0), d1)) - a) / span) * (r1 - r0);
}

const f1 = (v: number) => v.toFixed(1);

// --- dose-response -------------------------------------------------------------------------

export interface DoseBox { left: number; right: number; top: number; bottom: number }
export interface DoseModel {
  points: { rate: number; mean: number; low: number | null; high: number | null; x: number; y: number; yLow: number; yHigh: number }[];
  xTicks: { rate: number; x: number }[];
  yAxis: { max: number; ticks: number[] };
  y: (hz: number) => number;
  band: string;
  ec50: { x: number; label: string } | null;
}

export function doseModel(c: DoseChart, box: DoseBox): DoseModel {
  const rates = c.points.map((p) => p.rate);
  const x = logScale(rates[0], rates[rates.length - 1], box.left, box.right);
  const top = Math.max(0, ...c.points.map((p) => p.high ?? p.mean), ...c.points.map((p) => p.mean));
  const yAxis = niceTicks(top, 4);
  const y = linearScale(0, yAxis.max, box.bottom, box.top);
  const points = c.points.map((p) => ({
    ...p, x: x(p.rate), y: y(p.mean), yLow: y(p.low ?? p.mean), yHigh: y(p.high ?? p.mean),
  }));
  const hasBand = points.length > 1 && c.points.every((p) => p.low !== null && p.high !== null);
  const band = hasBand
    ? "M" + [...points.map((p) => `${p.x},${p.yHigh}`), ...[...points].reverse().map((p) => `${p.x},${p.yLow}`)].join("L") + "Z"
    : "";
  return {
    points, xTicks: points.map((p) => ({ rate: p.rate, x: p.x })), yAxis, y, band,
    ec50: c.ec50 === null ? null : { x: x(c.ec50), label: `EC50 ${Number(c.ec50.toFixed(1))} Hz` },
  };
}

export function doseTooltip(p: DosePoint, readout: string): string {
  const ci = p.low === null || p.high === null ? "no interval: 1 trial" : `95% CI ${f1(p.low)}–${f1(p.high)}`;
  return `${p.rate} Hz stimulus · ${readout} ${f1(p.mean)} Hz (${ci})`;
}

// --- screen --------------------------------------------------------------------------------

/** A symmetric axis around zero covering [min, max], with ticks at round values including 0. */
export function symmetricTicks(min: number, max: number): { min: number; max: number; ticks: number[] } {
  const { max: top, ticks } = niceTicks(Math.max(Math.abs(min), Math.abs(max)), 2);
  const neg = ticks.filter((t) => t > 0).map((t) => -t).reverse();
  return { min: -top, max: top, ticks: [...neg, 0, ...ticks.filter((t) => t > 0)] };
}

export interface ScreenBox { left: number; right: number; top: number; rowPx: number }
export interface ScreenModel {
  rows: { label: string; row: number; x: number; xLow: number; xHigh: number; y: number; filled: boolean; simulationId: string; tooltip: string }[];
  zeroX: number;
  xTicks: { value: number; x: number }[];
  height: number;
  /** y of the x axis line */
  axisY: number;
}

export function screenTooltip(r: ScreenRow): string {
  const ci = r.low === null || r.high === null ? "" : ` (95% CI ${f1(r.low)} – ${f1(r.high)})`;
  const verdict = r.p === null ? "no test (needs 2+ trials)"
    : `${r.p < 0.001 ? "p<0.001" : `p ${r.p}`} · ${r.significant ? "significant after correction" : "not significant"}`;
  return `${r.label} · row ${r.row} · Δ ${f1(r.delta)} Hz${ci} · ${verdict} · click to open its map`;
}

export function screenModel(c: ScreenChart, box: ScreenBox): ScreenModel {
  const lows = c.rows.map((r) => r.low ?? r.delta);
  const highs = c.rows.map((r) => r.high ?? r.delta);
  const axis = symmetricTicks(Math.min(0, ...lows), Math.max(0, ...highs));
  const x = linearScale(axis.min, axis.max, box.left, box.right);
  const rows = c.rows.map((r, i) => ({
    label: r.label, row: r.row, x: x(r.delta), xLow: x(r.low ?? r.delta), xHigh: x(r.high ?? r.delta),
    y: box.top + i * box.rowPx + box.rowPx / 2, filled: r.significant, simulationId: r.simulationId, tooltip: screenTooltip(r),
  }));
  const axisY = box.top + c.rows.length * box.rowPx;
  return { rows, zeroX: x(0), xTicks: axis.ticks.map((value) => ({ value, x: x(value) })), height: axisY + 40, axisY };
}

// --- progress ------------------------------------------------------------------------------

export interface ExperimentProgress { experiment: string; done: number; total: number }

export function parseProgress(d: unknown): ExperimentProgress | null {
  if (typeof d !== "object" || d === null) return null;
  const o = d as Record<string, unknown>;
  const int = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v) && v >= 0;
  if (typeof o.experiment !== "string" || !int(o.done) || !int(o.total) || o.done > o.total) return null;
  return { experiment: o.experiment, done: o.done, total: o.total };
}

const SHORT: Record<string, string> = { dose_response: "dose", silencing_screen: "screen" };

/** "screen · 3/8" */
export function formatExperimentProgress(p: ExperimentProgress): string {
  return `${SHORT[p.experiment] ?? p.experiment} · ${p.done}/${p.total}`;
}
