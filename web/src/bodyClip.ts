/** Physics walking clip (public/fly-walk.json, written by scripts/export_flygym_walk.py): per-body transforms of the
 *  leg bodies sampled from a NeuroMechFly v2 (flygym) walking simulation, relative to the thorax.
 *  Pure parsing and maths (no DOM, no three). Output: for each body the 4x4 D_body(t) = M_body(t) * M_body(neutral)^-1
 *  in the viewer frame (mm, Three axes), the rigid motion that takes the body's mesh from the neutral pose, in which
 *  the mesh is assembled, to its pose at time t. */
import type { Matrix16 } from "./bodyRig";

type Vec3 = [number, number, number];

/** Parsed clip. Positions are mm and rotations unit quaternions (w, x, y, z), both already in the viewer's axes. */
export interface WalkClip {
  bodies: string[];
  nFrames: number;
  sampleRateHz: number;
  /** nFrames / sampleRateHz: the loop length */
  durationS: number;
  /** thorax origin of the mesh frame (mm, viewer axes): the point the thorax-relative motion is about */
  thoraxOrigin: Vec3;
  /** frame-major: [frame][body][xyz] */
  pos: Float64Array;
  /** frame-major: [frame][body][wxyz] */
  quat: Float64Array;
  neutralPos: Float64Array;
  neutralQuat: Float64Array;
  /** per-leg adhesion flags (1 = the controller's stance command), frame-major [frame][leg]; empty when absent */
  adhesion: Uint8Array;
  legs: string[];
  /** true (the default, the committed walking demo): the clip repeats; false (a brain-driven run): it plays once and holds its last frame */
  loop: boolean;
  /** per-frame 1 = the drive said "escape" (legs stopped, wings raised); empty when absent */
  escape: Uint8Array;
  /** whole-body motion of the thorax in the world, viewer axes and mm, relative to the first frame; null when absent */
  world: { pos: Float64Array; yawRad: Float64Array } | null;
  /** flight wing angles, frame-major [frame][6] (degrees, flybody convention, left then right), from a live chunk; absent otherwise */
  wings?: Float64Array;
}

/** Root transform of the whole fly at one time: `matrix` (column-major 4x4, viewer frame, mm) carries the mesh from the
 *  start pose to its place in the world; `offset` is the thorax's translation (viewer axes, mm), for camera and floor. */
export interface WorldPose { matrix: Matrix16; offset: Vec3 }

const FORMAT = "fly-walk/1";
const UM_TO_MM = 1 / 1000;
const NUMBERS_PER_BODY = 7;

const isFiniteNumber = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isNumberArray = (v: unknown, n: number): v is number[] => Array.isArray(v) && v.length === n && v.every(isFiniteNumber);

/** Fly frame (x anterior, y left, z dorsal) -> viewer (X, Y, Z) = (y, z, x): the same cyclic permutation as bodyData. */
const posToViewer = (x: number, y: number, z: number): Vec3 => [y * UM_TO_MM, z * UM_TO_MM, x * UM_TO_MM];

function normalised(q: number[], scale: number, what: string): [number, number, number, number] {
  const w = q[0] / scale, x = q[1] / scale, y = q[2] / scale, z = q[3] / scale;
  const len = Math.hypot(w, x, y, z);
  if (!(len > 1e-6)) throw new Error(`fly walk clip: ${what} is a zero quaternion`);
  // a vector part is a vector: it permutes like a position; the scalar part is unchanged
  return [w / len, y / len, z / len, x / len];
}

/** Validate and convert the JSON. Throws a clear Error on anything malformed. */
export function parseWalkClip(raw: unknown): WalkClip {
  if (typeof raw !== "object" || raw === null) throw new Error("fly walk clip: not an object");
  const r = raw as Record<string, unknown>;
  if (r.format !== FORMAT) throw new Error(`fly walk clip: unsupported format ${String(r.format)} (expected ${FORMAT})`);
  const scale = r.quat_scale;
  if (!isFiniteNumber(scale) || scale <= 0) throw new Error("fly walk clip: bad quat_scale");
  const rate = r.sample_rate_hz;
  if (!isFiniteNumber(rate) || rate <= 0) throw new Error("fly walk clip: bad sample_rate_hz");
  const origin = r.thorax_origin_um;
  if (!isNumberArray(origin, 3)) throw new Error("fly walk clip: bad thorax_origin_um");
  const bodies = r.bodies;
  if (!Array.isArray(bodies) || bodies.length === 0 || !bodies.every((b) => typeof b === "string") || new Set(bodies).size !== bodies.length) {
    throw new Error("fly walk clip: bodies must be a non-empty list of unique names");
  }
  const nb = bodies.length;
  const frames = r.frames;
  if (!Array.isArray(frames) || frames.length === 0 || r.n_frames !== frames.length) throw new Error("fly walk clip: n_frames does not match frames");
  const neutral = r.neutral as { pos_um?: unknown; quat?: unknown } | undefined;
  if (!neutral || !Array.isArray(neutral.pos_um) || !Array.isArray(neutral.quat) || neutral.pos_um.length !== nb || neutral.quat.length !== nb) {
    throw new Error("fly walk clip: neutral needs one position and one quaternion per body");
  }

  const nFrames = frames.length;
  const pos = new Float64Array(nFrames * nb * 3);
  const quat = new Float64Array(nFrames * nb * 4);
  for (let f = 0; f < nFrames; f++) {
    const row = frames[f];
    if (!Array.isArray(row) || row.length !== nb) throw new Error(`fly walk clip: frame ${f} does not have ${nb} bodies`);
    for (let b = 0; b < nb; b++) {
      const v = row[b];
      if (!isNumberArray(v, NUMBERS_PER_BODY)) throw new Error(`fly walk clip: frame ${f} body ${String(bodies[b])} is not ${NUMBERS_PER_BODY} finite numbers`);
      pos.set(posToViewer(v[0], v[1], v[2]), (f * nb + b) * 3);
      quat.set(normalised([v[3], v[4], v[5], v[6]], scale, `frame ${f} body ${String(bodies[b])}`), (f * nb + b) * 4);
    }
  }
  const neutralPos = new Float64Array(nb * 3);
  const neutralQuat = new Float64Array(nb * 4);
  for (let b = 0; b < nb; b++) {
    const p = neutral.pos_um[b], q = neutral.quat[b];
    if (!isNumberArray(p, 3) || !isNumberArray(q, 4)) throw new Error(`fly walk clip: neutral of ${String(bodies[b])} is malformed`);
    neutralPos.set(posToViewer(p[0], p[1], p[2]), b * 3);
    neutralQuat.set(normalised(q, scale, `neutral ${String(bodies[b])}`), b * 4);
  }

  // optional per-leg flags (not needed to draw); anything malformed is simply left out
  const legs = Array.isArray(r.legs) && r.legs.every((l) => typeof l === "string") ? (r.legs as string[]) : [];
  let adhesion = new Uint8Array(0);
  const adh = r.adhesion;
  if (legs.length > 0 && Array.isArray(adh) && adh.length === nFrames && adh.every((row) => isNumberArray(row, legs.length))) {
    adhesion = Uint8Array.from((adh as number[][]).flat().map((v) => (v ? 1 : 0)));
  }
  const loop = typeof r.loop === "boolean" ? r.loop : true;
  const esc = r.escape;
  const escape = Array.isArray(esc) && esc.length === nFrames && esc.every(isFiniteNumber) ? Uint8Array.from((esc as number[]).map((v) => (v ? 1 : 0))) : new Uint8Array(0);
  return {
    bodies: bodies as string[],
    nFrames,
    sampleRateHz: rate,
    durationS: nFrames / rate,
    thoraxOrigin: posToViewer(origin[0], origin[1], origin[2]),
    pos, quat, neutralPos, neutralQuat, adhesion, legs, loop, escape, world: parseWorld(r.thorax_world, nFrames),
  };
}

/** thorax_world {pos_um, yaw_deg} (flygym world frame: x forward, y left, z up; x and y relative to the start, z absolute)
 *  -> viewer axes in mm, all relative to frame 0. Anything malformed means "no world motion". */
function parseWorld(raw: unknown, nFrames: number): WalkClip["world"] {
  if (typeof raw !== "object" || raw === null) return null;
  const w = raw as { pos_um?: unknown; yaw_deg?: unknown };
  if (!Array.isArray(w.pos_um) || w.pos_um.length !== nFrames || !w.pos_um.every((p) => isNumberArray(p, 3))) return null;
  if (!Array.isArray(w.yaw_deg) || w.yaw_deg.length !== nFrames || !w.yaw_deg.every(isFiniteNumber)) return null;
  const p0 = w.pos_um[0] as number[];
  const pos = new Float64Array(nFrames * 3);
  const yawRad = new Float64Array(nFrames);
  for (let f = 0; f < nFrames; f++) {
    const p = w.pos_um[f] as number[];
    pos.set(posToViewer(p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]), f * 3);
    yawRad[f] = (((w.yaw_deg as number[])[f] - (w.yaw_deg as number[])[0]) * Math.PI) / 180;
  }
  return { pos, yawRad };
}

/** Where time t (s) falls in the clip: frames i and j and the weight a of j. Looping wraps (the frame after the last is
 *  the first); otherwise t is clamped, so the last frame is held. Non-finite t falls back to the first frame. */
function locate(clip: WalkClip, t: number): { i: number; j: number; a: number } {
  const n = clip.nFrames;
  if (!clip.loop) {
    const u = Number.isFinite(t) ? Math.min(Math.max(t * clip.sampleRateHz, 0), n - 1) : 0;
    const i = Math.min(Math.floor(u), n - 1);
    return { i, j: Math.min(i + 1, n - 1), a: u - i };
  }
  let u = Number.isFinite(t) ? (t * clip.sampleRateHz) % n : 0;
  if (u < 0) u += n;
  const i = Math.min(Math.floor(u), n - 1);
  return { i, j: (i + 1) % n, a: u - i };
}

/** The six flight wing angles of the frame at time t, or null when the clip carries none (all zeros counts as none). */
export function wingsAt(clip: WalkClip, t: number): number[] | null {
  if (!clip.wings || clip.wings.length !== clip.nFrames * 6) return null;
  const i = locate(clip, t).i;
  const w = Array.from(clip.wings.subarray(i * 6, i * 6 + 6));
  return w.some((v) => v !== 0) ? w : null;
}

/** The escape flag of the frame at time t (false when the clip carries none). */
export function escapeAt(clip: WalkClip, t: number): boolean {
  if (clip.escape.length === 0) return false;
  return clip.escape[locate(clip, t).i] === 1;
}

/** The whole-fly root transform at time t, or null when the clip has no thorax_world. Rotation is about the vertical
 *  (viewer Y) axis through the thorax origin; yaw is positive to the left, which is a positive turn about +Y in viewer
 *  axes (fly forward = +Z turns toward fly left = +X). M = T(origin + p) Ry(yaw) T(-origin). */
export function worldPose(clip: WalkClip, t: number): WorldPose | null {
  const w = clip.world;
  if (!w) return null;
  const { i, j, a } = locate(clip, t);
  const p: Vec3 = [0, 0, 0];
  for (let k = 0; k < 3; k++) p[k] = (1 - a) * w.pos[i * 3 + k] + a * w.pos[j * 3 + k];
  const yaw = (1 - a) * w.yawRad[i] + a * w.yawRad[j];
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const T = clip.thoraxOrigin;
  const tx = T[0] + p[0] - (c * T[0] + s * T[2]);
  const tz = T[2] + p[2] - (-s * T[0] + c * T[2]);
  return { matrix: [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, tx, p[1], tz, 1], offset: p };
}

/** Row-major 3x3 of a unit quaternion (w, x, y, z). */
function rotationOf(q: ArrayLike<number>, at: number, out: Float64Array): void {
  const w = q[at], x = q[at + 1], y = q[at + 2], z = q[at + 3];
  out[0] = 1 - 2 * (y * y + z * z); out[1] = 2 * (x * y - z * w); out[2] = 2 * (x * z + y * w);
  out[3] = 2 * (x * y + z * w); out[4] = 1 - 2 * (x * x + z * z); out[5] = 2 * (y * z - x * w);
  out[6] = 2 * (x * z - y * w); out[7] = 2 * (y * z + x * w); out[8] = 1 - 2 * (x * x + y * y);
}

/** Spherical interpolation of the unit quaternions a (frame i) and b (frame j) of body `b`, the short way round. */
function slerp(clip: WalkClip, i: number, j: number, body: number, t: number, out: Float64Array): void {
  const nb = clip.bodies.length;
  const ai = (i * nb + body) * 4, bi = (j * nb + body) * 4;
  const q = clip.quat;
  let dot = q[ai] * q[bi] + q[ai + 1] * q[bi + 1] + q[ai + 2] * q[bi + 2] + q[ai + 3] * q[bi + 3];
  const sign = dot < 0 ? -1 : 1;
  dot = Math.abs(dot);
  let wa: number, wb: number;
  if (dot > 0.9995) { wa = 1 - t; wb = t; } // nearly parallel: a plain lerp is exact enough and avoids 0 / 0
  else {
    const theta = Math.acos(dot), s = Math.sin(theta);
    wa = Math.sin((1 - t) * theta) / s;
    wb = Math.sin(t * theta) / s;
  }
  let len = 0;
  for (let k = 0; k < 4; k++) { out[k] = wa * q[ai + k] + wb * sign * q[bi + k]; len += out[k] * out[k]; }
  len = Math.sqrt(len) || 1;
  for (let k = 0; k < 4; k++) out[k] /= len;
}

/** D_body(t) for every body, as column-major 4x4 matrices in the viewer frame (mm). `t` is in seconds and loops
 *  over `durationS` (negative and non-finite times wrap / fall back to the first frame); a clip with loop:false is clamped instead and holds its last frame; the frame after the last is
 *  the first. D = M(t) * M(neutral)^-1 acts in the thorax frame, whose origin sits at `thoraxOrigin` of the mesh frame,
 *  so the matrix is re-expressed about that point: x' = Rd x + (td + T - Rd T). */
export function clipMatrices(clip: WalkClip, t: number): Map<string, Matrix16> {
  const nb = clip.bodies.length;
  const { i, j, a } = locate(clip, t);
  const out = new Map<string, Matrix16>();
  const q = new Float64Array(4), rt = new Float64Array(9), rn = new Float64Array(9);
  const T = clip.thoraxOrigin;
  for (let b = 0; b < nb; b++) {
    slerp(clip, i, j, b, a, q);
    rotationOf(q, 0, rt);
    rotationOf(clip.neutralQuat, b * 4, rn);
    // Rd = Rt * Rn^T
    const rd = new Array<number>(9);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) rd[r * 3 + c] = rt[r * 3] * rn[c * 3] + rt[r * 3 + 1] * rn[c * 3 + 1] + rt[r * 3 + 2] * rn[c * 3 + 2];
    const p = [0, 0, 0];
    for (let k = 0; k < 3; k++) p[k] = (1 - a) * clip.pos[(i * nb + b) * 3 + k] + a * clip.pos[(j * nb + b) * 3 + k];
    const pn = [clip.neutralPos[b * 3], clip.neutralPos[b * 3 + 1], clip.neutralPos[b * 3 + 2]];
    const tr = [0, 0, 0];
    for (let r = 0; r < 3; r++) {
      const rdPn = rd[r * 3] * pn[0] + rd[r * 3 + 1] * pn[1] + rd[r * 3 + 2] * pn[2];
      const rdT = rd[r * 3] * T[0] + rd[r * 3 + 1] * T[1] + rd[r * 3 + 2] * T[2];
      tr[r] = p[r] - rdPn + T[r] - rdT;
    }
    out.set(clip.bodies[b], [rd[0], rd[3], rd[6], 0, rd[1], rd[4], rd[7], 0, rd[2], rd[5], rd[8], 0, tr[0], tr[1], tr[2], 1]);
  }
  return out;
}

/** The clip is a static file in web/public, served from the site root (not under /api). */
export async function loadWalkClip(fetchFn: typeof fetch = fetch, base = "/"): Promise<WalkClip> {
  const res = await fetchFn(`${base}fly-walk.json`);
  if (!res.ok) throw new Error(`could not load fly-walk.json: HTTP ${res.status}`);
  return parseWalkClip(await res.json());
}
