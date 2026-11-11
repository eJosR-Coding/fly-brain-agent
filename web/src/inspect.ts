/** Pure logic for click-to-inspect: picking, click-vs-drag, the neuron card model and its placement. */

export interface Hit { index: number; distanceToRay: number; distance: number }

/** The hit closest to the pointer ray (ties: nearer the camera, then lower index); null when nothing was hit. */
export function pickNearest(hits: readonly Hit[]): number | null {
  let best: Hit | null = null;
  for (const h of hits) {
    if (
      best === null ||
      h.distanceToRay < best.distanceToRay ||
      (h.distanceToRay === best.distanceToRay &&
        (h.distance < best.distance || (h.distance === best.distance && h.index < best.index)))
    ) {
      best = h;
    }
  }
  return best === null ? null : best.index;
}

export interface Pt { x: number; y: number }

/** A press and release that moved less than `thresholdPx` is a click; anything more was an orbit drag. */
export function isClick(down: Pt, up: Pt, thresholdPx = 5): boolean {
  return Math.hypot(up.x - down.x, up.y - down.y) <= thresholdPx;
}

/** World-space radius of `px` screen pixels at `distance` from a perspective camera (vertical fov in degrees). */
export function pickThreshold(px: number, fovDeg: number, distance: number, heightPx: number): number {
  return (px * 2 * Math.tan((fovDeg * Math.PI) / 360) * distance) / Math.max(heightPx, 1);
}

export interface Partner { row: number; cellType: string | null; superClass: string | null; synapses: number }
export interface NeuronCard {
  row: number;
  rootId: string;
  cellType: string | null;
  cellClass: string | null;
  superClass: string | null;
  side: string | null;
  topNt: string | null;
  position: [number, number, number];
  region: string | null;
  /** one or two plain-English sentences from the server; null when it sent none */
  plain: string | null;
  /** null when the server has no connectome */
  partners: { inputs: Partner[]; outputs: Partner[]; synapsesIn: number; synapsesOut: number } | null;
}

const text = (v: unknown): string | null => (typeof v === "string" && v !== "" ? v : null);
const count = (v: unknown): number => (typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : 0);

function partnerList(v: unknown): Partner[] {
  if (!Array.isArray(v)) return [];
  const out: Partner[] = [];
  for (const p of v) {
    if (typeof p !== "object" || p === null) continue;
    const r = p as Record<string, unknown>;
    if (typeof r.row !== "number" || !Number.isInteger(r.row) || r.row < 0) continue;
    if (typeof r.synapses !== "number" || !Number.isFinite(r.synapses)) continue;
    out.push({ row: r.row, cellType: text(r.cell_type), superClass: text(r.super_class), synapses: r.synapses });
  }
  return out;
}

/** Check a GET /neurons/{row} payload from the wire: the typed card, or why it is invalid. */
export function parseNeuronCard(d: unknown, pointCount: number | null): NeuronCard | string {
  if (typeof d !== "object" || d === null) return "neuron card is not an object";
  const r = d as Record<string, unknown>;
  if (typeof r.row !== "number" || !Number.isInteger(r.row) || r.row < 0 || (pointCount !== null && r.row >= pointCount)) {
    return `bad row ${JSON.stringify(r.row)}`;
  }
  if (typeof r.root_id !== "string" || !/^\d+$/.test(r.root_id)) return "bad root_id";
  const pos = r.position_um;
  if (!Array.isArray(pos) || pos.length !== 3 || !pos.every((n) => typeof n === "number" && Number.isFinite(n))) {
    return "bad position_um";
  }
  return {
    row: r.row,
    rootId: r.root_id,
    cellType: text(r.cell_type),
    cellClass: text(r.cell_class),
    superClass: text(r.super_class),
    side: text(r.side),
    topNt: text(r.top_nt),
    position: [pos[0], pos[1], pos[2]],
    region: text(r.region),
    plain: text(r.plain),
    partners:
      r.partners === true
        ? { inputs: partnerList(r.inputs), outputs: partnerList(r.outputs), synapsesIn: count(r.synapses_in), synapsesOut: count(r.synapses_out) }
        : null,
  };
}

export interface PartnerLine { row: number; text: string; syn: string }
export interface CardView {
  title: string;
  /** the plain-English line shown under the title */
  plain: string | null;
  rows: { label: string; value: string }[];
  inputs: PartnerLine[];
  outputs: PartnerLine[];
  /** shown instead of the partner lists when the server cannot provide them */
  partnersNote: string | null;
}

/** Hover text for the technical labels on the card (plain words, nothing removed). */
export const LABEL_TIPS: Record<string, string> = {
  "root id": "The FlyWire identifier of this neuron's 3D reconstruction; it is unique and stable in this data release.",
  class: "The FlyWire cell class: a coarse family this neuron was assigned to.",
  region: "The broad brain region where the cell body (soma) sits.",
  position: "Where the neuron sits in the brain volume, in micrometers (x, y, z).",
  synapses: "Total synapses this neuron receives (in) and sends (out), counting every partner above the connectome threshold.",
};
export const PARTNER_TIPS: Record<string, string> = {
  "top inputs": "The neurons that send the most synapses to this one, with the synapse count.",
  "top outputs": "The neurons that receive the most synapses from this one, with the synapse count.",
};

const UNANNOTATED = "unannotated";
const fmt = (n: number) => n.toLocaleString("en-US");

/** "PAM05 · central · left · dopamine": what the neuron is, at a glance. */
export function cardView(c: NeuronCard): CardView {
  const title = [c.cellType ?? UNANNOTATED, c.superClass, c.side, c.topNt].filter((p) => p !== null).join(" · ");
  const rows = [{ label: "root id", value: c.rootId }];
  if (c.cellClass) rows.push({ label: "class", value: c.cellClass });
  if (c.region) rows.push({ label: "region", value: c.region });
  rows.push({ label: "position", value: `${c.position.map((n) => n.toFixed(1)).join(", ")} µm` });
  const line = (p: Partner): PartnerLine => ({
    row: p.row,
    text: [p.cellType ?? UNANNOTATED, p.superClass].filter((x) => x !== null).join(" · "),
    syn: `${fmt(p.synapses)} syn`,
  });
  if (c.partners === null) {
    return { title, plain: c.plain, rows, inputs: [], outputs: [], partnersNote: "partners need the connectome (not loaded on this server)" };
  }
  rows.push({ label: "synapses", value: `${fmt(c.partners.synapsesIn)} in · ${fmt(c.partners.synapsesOut)} out` });
  return { title, plain: c.plain, rows, inputs: c.partners.inputs.map(line), outputs: c.partners.outputs.map(line), partnersNote: null };
}

/**
 * Top-left corner for a card of `size` next to `anchor`: right of it by `gap`, flipped to the left when
 * that would overflow `bounds`, vertically centered on it and kept `margin` inside the bounds.
 */
export function placeCard(
  anchor: Pt,
  size: { w: number; h: number },
  bounds: { w: number; h: number },
  o: { gap: number; margin: number },
): { left: number; top: number } {
  let left = anchor.x + o.gap;
  if (left + size.w > bounds.w - o.margin) left = anchor.x - o.gap - size.w;
  left = Math.max(o.margin, Math.min(left, bounds.w - o.margin - size.w));
  const top = Math.max(o.margin, Math.min(anchor.y - size.h / 2, bounds.h - o.margin - size.h));
  return { left, top };
}
