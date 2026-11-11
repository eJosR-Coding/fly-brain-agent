/** Pure logic for drawing find_path results as cables: payload validation, hop geometry, widths and labels. */
import type { Vec3 } from "./brainData";

export interface PathPayload { selectionId: string; rows: number[]; synapses: number[]; bottleneck: number }

const count = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v) && v >= 0;

/** Check a `path` SSE payload: rows in range, one synapse count per hop. Returns the typed path, or why it is invalid. */
export function parsePathEvent(d: unknown, pointCount: number | null): PathPayload | string {
  if (typeof d !== "object" || d === null || Array.isArray(d)) return "path event is not an object";
  const r = d as Record<string, unknown>;
  if (!Array.isArray(r.rows) || r.rows.length < 2) return "rows needs at least two neurons";
  for (const row of r.rows) {
    if (typeof row !== "number" || !Number.isInteger(row) || row < 0 || (pointCount !== null && row >= pointCount)) {
      return `bad row ${JSON.stringify(row)}`;
    }
  }
  if (!Array.isArray(r.synapses) || r.synapses.length !== r.rows.length - 1 || !r.synapses.every(count)) {
    return "synapses must hold one non-negative number per hop";
  }
  if (!count(r.bottleneck)) return "bottleneck is not a number";
  return {
    selectionId: typeof r.selection_id === "string" ? r.selection_id : "",
    rows: r.rows as number[],
    synapses: r.synapses as number[],
    bottleneck: r.bottleneck,
  };
}

/** Cable width in px: a fine thread (1 px) that thickens subtly with the log of the hop's synapses, up to 2.5 px. */
export function hopWidth(synapses: number): number {
  const s = Math.max(1, synapses);
  return Math.min(2.5, Math.max(1, 0.9 + 0.2 * Math.log(s)));
}

export const hopLabel = (synapses: number): string => `${synapses.toLocaleString("en-US")} syn`;

/** "path · 3 hops · bottleneck 5 syn". */
export function pathLegend(p: PathPayload): string {
  const hops = p.rows.length - 1;
  return `path · ${hops} hop${hops === 1 ? "" : "s"} · bottleneck ${hopLabel(p.bottleneck)}`;
}

export interface Hop { from: Vec3; to: Vec3; mid: Vec3; synapses: number; width: number; label: string }

const at = (positions: ArrayLike<number>, row: number): Vec3 => [positions[row * 3], positions[row * 3 + 1], positions[row * 3 + 2]];

/** The route's neuron positions in order (raw data frame, micrometers). */
export function routeVertices(p: PathPayload, positions: ArrayLike<number>): Vec3[] {
  return p.rows.map((r) => at(positions, r));
}

/** One cable segment per consecutive pair of neurons. */
export function buildHops(p: PathPayload, positions: ArrayLike<number>): Hop[] {
  const v = routeVertices(p, positions);
  return p.synapses.map((synapses, i) => ({
    from: v[i],
    to: v[i + 1],
    mid: [(v[i][0] + v[i + 1][0]) / 2, (v[i][1] + v[i + 1][1]) / 2, (v[i][2] + v[i + 1][2]) / 2],
    synapses,
    width: hopWidth(synapses),
    label: hopLabel(synapses),
  }));
}

/** The point at fraction `t` (0..1, clamped) of the polyline's total length: where the traveling light is. */
export function pointAlong(v: readonly Vec3[], t: number): Vec3 {
  if (v.length === 0) return [0, 0, 0];
  const lens: number[] = [];
  let total = 0;
  for (let i = 1; i < v.length; i++) {
    const l = Math.hypot(v[i][0] - v[i - 1][0], v[i][1] - v[i - 1][1], v[i][2] - v[i - 1][2]);
    lens.push(l);
    total += l;
  }
  if (total === 0) return [...v[0]] as Vec3;
  let want = Math.min(1, Math.max(0, t)) * total;
  for (let i = 0; i < lens.length; i++) {
    if (want <= lens[i] || i === lens.length - 1) {
      const f = lens[i] === 0 ? 0 : Math.min(1, want / lens[i]);
      return [0, 1, 2].map((k) => v[i][k] + (v[i + 1][k] - v[i][k]) * f) as Vec3;
    }
    want -= lens[i];
  }
  return [...v[v.length - 1]] as Vec3;
}
