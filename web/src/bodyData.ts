/** Fly body asset: line segments of the NeuroMechFly v2 micro-CT mesh, packaged as /fly-body.json + /fly-body.bin.
 *  Pure decoding (no DOM, no three). Bin: per segment two endpoints, each uint16 x,y,z little-endian (12 bytes),
 *  quantized per axis over meta.bounds (um = min + q / 65535 * (max - min)). */

import type { JointId } from "./bodyPose";

export type Vec3 = [number, number, number];

export interface BodyPartMeta {
  name: string;
  group: string;
  side: string;
  segment_start: number;
  segment_count: number;
}

/** The optional `mesh` block of fly-body.json: layout of fly-body-mesh.bin (vertices uint16 x3, then triangle indices). */
export interface MeshMeta {
  file: string;
  vertex_count: number;
  triangle_count: number;
  index_dtype: "uint16" | "uint32";
  bytes: number;
  /** triangle range of each part inside the concatenated triangle list (part order) */
  parts?: Array<{ name: string; triangle_start: number; triangle_count: number }>;
}

/** One entry of the `rig` block (fly frame, micrometres), as written by the asset build. */
export interface RigJointMeta {
  id: string;
  parts: string[];
  pivot_um: Vec3;
  axis: Vec3;
  rest_deg?: number;
  range_deg?: [number, number];
  source?: string;
}

/** A joint in the viewer frame: rotation by an angle about `axis` (unit) through `pivot` (mm, Three axes). */
export interface RigJoint {
  id: JointId;
  parts: string[];
  pivot: Vec3;
  axis: Vec3;
}

/** A contiguous range inside a per-part list (segments of the lines, or triangles of the mesh). */
export interface PartRange {
  name: string;
  start: number;
  count: number;
}

export interface BodyMesh {
  /** xyz per vertex, millimetres, Three.js axes */
  positions: Float32Array;
  /** 3 indices per triangle */
  indices: Uint16Array | Uint32Array;
}

export interface BodyMeta {
  /** [min, max] in micrometres, fly frame: x anterior(+), y left(+), z dorsal(+) */
  bounds: [Vec3, Vec3];
  head_center_um: Vec3;
  parts: BodyPartMeta[];
  mesh?: MeshMeta;
  rig?: { units?: string; frame?: string; joints: RigJointMeta[] };
}

export interface BodyPart {
  name: string;
  group: string;
  side: string;
  start: number;
  count: number;
}

export interface ParsedBody {
  /** xyz per endpoint, 2 endpoints per segment, millimetres, Three.js axes */
  positions: Float32Array;
  segmentCount: number;
  bounds: { min: Vec3; max: Vec3 };
  center: Vec3;
  /** mean of all segment endpoints (the visual centre of mass: the orbit target) */
  centroid: Vec3;
  /** farthest vertex from the centroid, mm */
  radius: number;
  headCenter: Vec3;
  parts: BodyPart[];
  /** depth-only occluder surface for the hidden-line look; null when absent or failed to load */
  mesh: BodyMesh | null;
  /** movable joints (empty for a body without `rig`) */
  joints: RigJoint[];
  /** triangle range of each part in `mesh.indices` (empty when the json has no per-part mesh ranges) */
  meshParts: PartRange[];
}

const BYTES_PER_SEGMENT = 12;
const UM_TO_MM = 1 / 1000;

/** Fly frame (x anterior, y left, z dorsal) -> Three.js axes (X right of the viewer, Y up, Z toward the viewer):
 *  three = (y_fly, z_fly, x_fly). It is a cyclic permutation, so handedness is kept (x cross y = z maps to
 *  Z cross X = Y), and a fly facing the camera has its left on the viewer's right. */
const toThree = (x: number, y: number, z: number): Vec3 => [y, z, x];

export function parseBody(buffer: ArrayBuffer, meta: BodyMeta): ParsedBody {
  const covered = meta.parts.reduce((n, p) => n + p.segment_count, 0);
  if (buffer.byteLength !== covered * BYTES_PER_SEGMENT) {
    throw new Error(`fly body is ${buffer.byteLength} bytes, its parts describe ${covered} segments (needs ${covered * BYTES_PER_SEGMENT})`);
  }
  const segmentCount = covered;
  let next = 0;
  for (const p of meta.parts) {
    if (p.segment_start !== next) throw new Error(`fly body parts are not contiguous at ${p.name} (starts at ${p.segment_start}, expected ${next})`);
    next += p.segment_count;
  }
  const [lo, hi] = meta.bounds;
  const view = new DataView(buffer);
  const positions = new Float32Array(segmentCount * 6);
  for (let i = 0; i < segmentCount * 2; i++) {
    const q = (a: number) => view.getUint16((i * 3 + a) * 2, true);
    const fx = (lo[0] + (q(0) / 65535) * (hi[0] - lo[0])) * UM_TO_MM;
    const fy = (lo[1] + (q(1) / 65535) * (hi[1] - lo[1])) * UM_TO_MM;
    const fz = (lo[2] + (q(2) / 65535) * (hi[2] - lo[2])) * UM_TO_MM;
    positions.set(toThree(fx, fy, fz), i * 3);
  }
  const min = toThree(lo[0] * UM_TO_MM, lo[1] * UM_TO_MM, lo[2] * UM_TO_MM);
  const max = toThree(hi[0] * UM_TO_MM, hi[1] * UM_TO_MM, hi[2] * UM_TO_MM);
  const [hx, hy, hz] = meta.head_center_um;
  const mid = centroid(positions);
  return {
    positions,
    segmentCount,
    bounds: { min, max },
    center: [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2],
    centroid: mid,
    radius: boundingRadius(positions, mid),
    headCenter: toThree(hx * UM_TO_MM, hy * UM_TO_MM, hz * UM_TO_MM),
    parts: meta.parts.map((p) => ({ name: p.name, group: p.group, side: p.side, start: p.segment_start, count: p.segment_count })),
    mesh: null,
    joints: parseRig(meta),
    meshParts: (meta.mesh?.parts ?? []).map((p) => ({ name: p.name, start: p.triangle_start, count: p.triangle_count })),
  };
}

/** `rig` joints converted into the viewer frame. A joint that cannot be applied (unknown part, bad pivot or axis) is
 *  dropped with one warning: the body then simply has no such joint and stays at rest there. */
function parseRig(meta: BodyMeta): RigJoint[] {
  const known = new Set(meta.parts.map((p) => p.name));
  const out: RigJoint[] = [];
  const isVec3 = (v: unknown): v is Vec3 => Array.isArray(v) && v.length === 3 && v.every((n) => typeof n === "number" && Number.isFinite(n));
  for (const j of meta.rig?.joints ?? []) {
    const unknown = Array.isArray(j.parts) ? j.parts.filter((n) => !known.has(n)) : ["(no parts)"];
    if (unknown.length > 0) { console.warn(`fly body joint ${j.id} dropped: unknown part ${unknown.join(", ")}`); continue; }
    if (!isVec3(j.pivot_um) || !isVec3(j.axis)) { console.warn(`fly body joint ${j.id} dropped: malformed pivot or axis`); continue; }
    const axis = toThree(j.axis[0], j.axis[1], j.axis[2]);
    const len = Math.hypot(axis[0], axis[1], axis[2]);
    if (!(len > 1e-9)) { console.warn(`fly body joint ${j.id} dropped: zero axis`); continue; }
    out.push({
      id: j.id as JointId,
      parts: [...j.parts],
      pivot: toThree(j.pivot_um[0] * UM_TO_MM, j.pivot_um[1] * UM_TO_MM, j.pivot_um[2] * UM_TO_MM),
      axis: [axis[0] / len, axis[1] / len, axis[2] / len],
    });
  }
  return out;
}

/** fly-body-mesh.bin -> positions (mm, Three axes, same decode rule as the lines) and triangle indices. */
export function parseBodyMesh(buffer: ArrayBuffer, meta: BodyMeta): BodyMesh {
  const info = meta.mesh;
  if (!info) throw new Error("fly body json has no mesh block");
  if (info.index_dtype !== "uint16" && info.index_dtype !== "uint32") throw new Error(`unknown mesh index_dtype ${String(info.index_dtype)}`);
  const idxBytes = info.index_dtype === "uint16" ? 2 : 4;
  const need = info.vertex_count * 6 + info.triangle_count * 3 * idxBytes;
  if (buffer.byteLength !== need) {
    throw new Error(`fly body mesh is ${buffer.byteLength} bytes, its metadata describes ${info.vertex_count} vertices and ${info.triangle_count} triangles (needs ${need})`);
  }
  const [lo, hi] = meta.bounds;
  const view = new DataView(buffer);
  const positions = new Float32Array(info.vertex_count * 3);
  for (let i = 0; i < info.vertex_count; i++) {
    const q = (a: number) => view.getUint16((i * 3 + a) * 2, true);
    const fx = (lo[0] + (q(0) / 65535) * (hi[0] - lo[0])) * UM_TO_MM;
    const fy = (lo[1] + (q(1) / 65535) * (hi[1] - lo[1])) * UM_TO_MM;
    const fz = (lo[2] + (q(2) / 65535) * (hi[2] - lo[2])) * UM_TO_MM;
    positions.set(toThree(fx, fy, fz), i * 3);
  }
  // slice() copies into an aligned buffer: the index block starts at an arbitrary even offset
  const block = buffer.slice(info.vertex_count * 6);
  const indices = info.index_dtype === "uint16" ? new Uint16Array(block) : new Uint32Array(block);
  for (let i = 0; i < indices.length; i++) {
    if (indices[i] >= info.vertex_count) throw new Error(`fly body mesh index ${indices[i]} is out of range (${info.vertex_count} vertices)`);
  }
  return { positions, indices };
}

/** The body is a static file in web/public, served from the site root (not under /api). */
export async function loadBody(fetchFn: typeof fetch = fetch, base = "/"): Promise<ParsedBody> {
  const get = async (name: string) => {
    const res = await fetchFn(`${base}${name}`);
    if (!res.ok) throw new Error(`could not load ${name}: HTTP ${res.status}`);
    return res;
  };
  const [metaRes, binRes] = await Promise.all([get("fly-body.json"), get("fly-body.bin")]);
  const meta = (await metaRes.json()) as BodyMeta;
  const body = parseBody(await binRes.arrayBuffer(), meta);
  if (meta.mesh) {
    // the occluder is an enhancement: if it fails the body still draws, as plain lines
    try {
      body.mesh = parseBodyMesh(await (await get(meta.mesh.file)).arrayBuffer(), meta);
    } catch (e) {
      console.warn(`fly body mesh unavailable, drawing the lines only: ${(e as Error).message}`);
    }
  }
  return body;
}

/** Mean of all vertices (xyz triplets); [0, 0, 0] for an empty buffer. */
export function centroid(positions: ArrayLike<number>): Vec3 {
  const n = Math.floor(positions.length / 3);
  if (n === 0) return [0, 0, 0];
  let x = 0, y = 0, z = 0;
  for (let i = 0; i < n; i++) {
    x += positions[i * 3];
    y += positions[i * 3 + 1];
    z += positions[i * 3 + 2];
  }
  return [x / n, y / n, z / n];
}

/** Distance from `center` to the farthest vertex (0 for an empty buffer). */
export function boundingRadius(positions: ArrayLike<number>, center: Vec3): number {
  let r2 = 0;
  for (let i = 0; i + 2 < positions.length; i += 3) {
    const dx = positions[i] - center[0], dy = positions[i + 1] - center[1], dz = positions[i + 2] - center[2];
    r2 = Math.max(r2, dx * dx + dy * dy + dz * dz);
  }
  return Math.sqrt(r2);
}

/** Camera distance from `target` along the view direction (azimuth/elevation in degrees, as in body.ts) at which the
 *  vertices' projection reaches `fill` (0..1) of the limiting half-extent of the view. Exact for a perspective camera:
 *  a vertex at depth offset w toward the camera and lateral offset l needs d >= w + |l| / (fill * tan(half fov)).
 *
 *  Guards: `aspect` is floored at 0.01 and `fill` at 0.05 so a collapsed container or a zero fill cannot divide by
 *  zero; every vertex keeps the camera 1e-3 in front of it. Degenerate input: no vertices returns 1 (a harmless
 *  distance); vertices all on the target (zero radius) return 1e-3, the smallest distance that keeps them in front. */
export function viewFitDistance(
  positions: ArrayLike<number>,
  target: Vec3,
  azimuthDeg: number,
  elevationDeg: number,
  fovDeg: number,
  aspect: number,
  fill: number,
): number {
  const a = (azimuthDeg * Math.PI) / 180, e = (elevationDeg * Math.PI) / 180;
  const dir = [Math.sin(a) * Math.cos(e), Math.sin(e), Math.cos(a) * Math.cos(e)]; // target -> camera
  const right = [Math.cos(a), 0, -Math.sin(a)];
  const up = [dir[1] * right[2] - dir[2] * right[1], dir[2] * right[0] - dir[0] * right[2], dir[0] * right[1] - dir[1] * right[0]];
  const tanV = Math.tan((fovDeg * Math.PI) / 360);
  const tanH = tanV * Math.max(aspect, 0.01);
  const f = Math.max(fill, 0.05);
  let d = 0;
  for (let i = 0; i + 2 < positions.length; i += 3) {
    const px = positions[i] - target[0], py = positions[i + 1] - target[1], pz = positions[i + 2] - target[2];
    const w = px * dir[0] + py * dir[1] + pz * dir[2];
    const lx = Math.abs(px * right[0] + py * right[1] + pz * right[2]);
    const ly = Math.abs(px * up[0] + py * up[1] + pz * up[2]);
    d = Math.max(d, w + 1e-3, w + lx / (f * tanH), w + ly / (f * tanV)); // 1e-3: never at or behind the camera
  }
  return d > 0 ? d : 1;
}
