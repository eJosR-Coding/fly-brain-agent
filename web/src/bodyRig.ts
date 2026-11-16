/** Pure rig maths and the split of the body into per-joint pieces. No DOM, no three.
 *  Vertex data is never moved: a joint's group is rotated by a matrix, so positions stay the asset's own. */
import type { PartRange, BodyPart, Vec3 } from "./bodyData";

export type Matrix16 = number[];

/** Column-major 4x4 (three's layout) rotating by `angleRad` about the unit-normalised `axis` through `pivot`
 *  (Rodrigues; positive follows the right-hand rule about the axis): p' = R (p - pivot) + pivot. */
export function rotationMatrix(axis: ArrayLike<number>, angleRad: number, pivot: ArrayLike<number>): Matrix16 {
  const len = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  const x = axis[0] / len, y = axis[1] / len, z = axis[2] / len;
  const c = Math.cos(angleRad), s = Math.sin(angleRad), t = 1 - c;
  const r00 = t * x * x + c, r01 = t * x * y - s * z, r02 = t * x * z + s * y;
  const r10 = t * x * y + s * z, r11 = t * y * y + c, r12 = t * y * z - s * x;
  const r20 = t * x * z - s * y, r21 = t * y * z + s * x, r22 = t * z * z + c;
  const tx = pivot[0] - (r00 * pivot[0] + r01 * pivot[1] + r02 * pivot[2]);
  const ty = pivot[1] - (r10 * pivot[0] + r11 * pivot[1] + r12 * pivot[2]);
  const tz = pivot[2] - (r20 * pivot[0] + r21 * pivot[1] + r22 * pivot[2]);
  return [r00, r10, r20, 0, r01, r11, r21, 0, r02, r12, r22, 0, tx, ty, tz, 1];
}

export function applyMatrix(p: ArrayLike<number>, m: ArrayLike<number>): Vec3 {
  return [
    m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
    m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
    m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
  ];
}

export interface Piece {
  /** segment endpoints, xyz each, 2 per segment, in part order */
  lines: Float32Array;
  /** triangle indices into the SHARED mesh position array (null when there is no mesh), in part order */
  indices: Uint16Array | Uint32Array | null;
}

export interface SplitInput {
  parts: BodyPart[];
  meshParts: PartRange[];
  joints: Array<{ id: string; parts: string[] }>;
  positions: Float32Array;
  indices: Uint16Array | Uint32Array | null;
}

/** Each joint gets the lines and triangles of ITS parts; `rest` gets every part in no joint. A part claimed by two
 *  joints belongs to the first. Parts keep the asset order inside every piece. Vertex indices are not remapped. */
export function splitByJoint(input: SplitInput): { joints: Record<string, Piece>; rest: Piece } {
  const { parts, meshParts, joints, positions, indices } = input;
  const owner = new Map<string, string>();
  for (const j of joints) for (const n of j.parts) if (!owner.has(n)) owner.set(n, j.id);
  const lineBuckets = new Map<string, Float32Array[]>();
  const triBuckets = new Map<string, Array<Uint16Array | Uint32Array>>();
  const REST = "\0rest";
  const push = <T,>(m: Map<string, T[]>, k: string, v: T) => { const a = m.get(k); if (a) a.push(v); else m.set(k, [v]); };
  for (const p of parts) push(lineBuckets, owner.get(p.name) ?? REST, positions.slice(p.start * 6, (p.start + p.count) * 6));
  if (indices) for (const p of meshParts) push(triBuckets, owner.get(p.name) ?? REST, indices.slice(p.start * 3, (p.start + p.count) * 3));
  const join = (chunks: Float32Array[] | undefined) => {
    const out = new Float32Array((chunks ?? []).reduce((n, c) => n + c.length, 0));
    let at = 0;
    for (const c of chunks ?? []) { out.set(c, at); at += c.length; }
    return out;
  };
  const joinIdx = (chunks: Array<Uint16Array | Uint32Array> | undefined): Uint16Array | Uint32Array | null => {
    if (!indices) return null;
    const out = indices instanceof Uint16Array ? new Uint16Array((chunks ?? []).reduce((n, c) => n + c.length, 0)) : new Uint32Array((chunks ?? []).reduce((n, c) => n + c.length, 0));
    let at = 0;
    for (const c of chunks ?? []) { out.set(c, at); at += c.length; }
    return out;
  };
  const piece = (k: string): Piece => ({ lines: join(lineBuckets.get(k)), indices: joinIdx(triBuckets.get(k)) });
  const out: Record<string, Piece> = {};
  for (const j of joints) out[j.id] = piece(j.id);
  return { joints: out, rest: piece(REST) };
}
