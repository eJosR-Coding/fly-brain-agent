/**
 * Pure helpers for the anatomy context: label specs and camera presets built ONLY from
 * /brain/landmarks numbers (centroids, lamina direction, lobe radius). Nothing here touches
 * neuron data. All coordinates are micrometers in the raw data frame.
 */
import type { Vec3 } from "./brainData";

export interface Landmark { name: string; x: number; y: number; z: number; count: number }
export interface Side { lobe_center: Vec3; radius: number; lamina_direction: Vec3 | null }
export interface Landmarks { landmarks: Landmark[]; sides: { left?: Side; right?: Side } }
export type SideName = "left" | "right";

export interface LabelSpec { text: string; position: Vec3; kind: "side" | "top" }

/** Thin labels anchored to landmark centroids: the central brain goes above the cloud, the rest to its sides. */
export function labelSpecs(lm: Landmarks): LabelSpec[] {
  const out: LabelSpec[] = [];
  for (const m of lm.landmarks) {
    const position: Vec3 = [m.x, m.y, m.z];
    if (m.name === "central brain") out.push({ text: "central brain", position, kind: "top" });
    else if (m.name.startsWith("optic lobe ")) out.push({ text: "optic lobe", position, kind: "side" });
    else if (m.name.startsWith("lamina ")) out.push({ text: "lamina · eye input", position, kind: "side" });
  }
  return out;
}

export interface Box { minX: number; maxX: number; minY: number; maxY: number }

/** Bounds of a flat [x0, y0, x1, y1, ...] screen-space list; null when empty. */
export function screenBounds(xy: ArrayLike<number>): Box | null {
  if (xy.length < 2) return null;
  const box: Box = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity };
  for (let i = 0; i + 1 < xy.length; i += 2) {
    box.minX = Math.min(box.minX, xy[i]); box.maxX = Math.max(box.maxX, xy[i]);
    box.minY = Math.min(box.minY, xy[i + 1]); box.maxY = Math.max(box.maxY, xy[i + 1]);
  }
  return box;
}

export interface LabelInput { ax: number; ay: number; kind: LabelSpec["kind"] }
export interface LabelPlacement { side: "left" | "right" | "top"; dx: number; dy: number; length: number; angle: number }
export interface PlaceOptions { margin: number; spacing: number; width: number; height: number }

/**
 * Screen-space label layout. Text ends sit OUTSIDE the cloud's projected bounds (plus a margin), on the
 * screen side of the anchor relative to `centerX`; same-side labels are kept `spacing` apart. Returned
 * offsets are relative to each anchor, with the hairline leader's length and angle (degrees).
 */
export function placeLabels(items: LabelInput[], box: Box, centerX: number, o: PlaceOptions): LabelPlacement[] {
  const out: LabelPlacement[] = new Array(items.length);
  const sides: { left: number[]; right: number[] } = { left: [], right: [] };
  const edge = { left: box.minX - o.margin, right: box.maxX + o.margin, top: Math.max(box.minY - o.margin, o.spacing / 2) };
  const targets: { x: number; y: number; side: LabelPlacement["side"] }[] = new Array(items.length);
  items.forEach((it, i) => {
    if (it.kind === "top") { targets[i] = { x: it.ax, y: Math.min(edge.top, it.ay - 8), side: "top" }; return; }
    const side = it.ax < centerX ? "left" : "right";
    // an anchor already beyond the margin keeps its leader pointing outward
    const x = side === "left" ? Math.min(edge.left, it.ax - 8) : Math.max(edge.right, it.ax + 8);
    targets[i] = { x, y: it.ay, side };
    sides[side].push(i);
  });
  for (const idx of [sides.left, sides.right]) {
    idx.sort((a, b) => targets[a].y - targets[b].y);
    let prev = -Infinity;
    for (const i of idx) {
      targets[i].y = Math.max(targets[i].y, o.spacing / 2, prev + o.spacing);
      prev = targets[i].y;
    }
  }
  items.forEach((it, i) => {
    const t = targets[i];
    const dx = t.x - it.ax, dy = t.y - it.ay;
    out[i] = { side: t.side, dx, dy, length: Math.hypot(dx, dy), angle: (Math.atan2(dy, dx) * 180) / Math.PI };
  });
  return out;
}

/** Camera position around the origin: azimuth turns away from the -z side (where the eyes face), elevation rises. */
export function homeCamera(distance: number, azimuthDeg: number, elevationDeg: number): Vec3 {
  const az = (azimuthDeg * Math.PI) / 180, el = (elevationDeg * Math.PI) / 180;
  return [distance * Math.cos(el) * Math.sin(az), distance * Math.sin(el), -distance * Math.cos(el) * Math.cos(az)];
}
