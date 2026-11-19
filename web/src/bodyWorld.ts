/** Pure helpers of the body's little world: the wings of an escape, the floor grid and the camera follow.
 *  No DOM, no three. Viewer axes: X = fly left, Y = up, Z = fly forward, mm. */
import { MAPPING, type JointId } from "./bodyPose";

type Vec3 = [number, number, number];

/** Wing raise for an escape frame: the same full angle the recording's DNp01 mapping uses (illustrative, no takeoff model). */
export const WING_RAISE_DEG = MAPPING.dnp01.maxDeg;

/** Joint angles for one frame of a body clip: both wings up on an escape frame, rest otherwise. */
export function escapePose(escape: boolean): Partial<Record<JointId, number>> {
  return escape ? { wing_l: WING_RAISE_DEG, wing_r: WING_RAISE_DEG } : {};
}

/** Where the floor grid sits for a fly at `offset`: the nearest whole cell at or below it in X and Z, so the lines stay
 *  fixed in the world while the fly walks over them. Returns [x, z]. */
export function gridSnap(offset: ArrayLike<number>, cell: number): [number, number] {
  const f = (v: number) => Math.floor(v / cell) * cell + 0; // + 0 turns -0 into 0
  return [f(offset[0]), f(offset[2])];
}

/** Line segments (xyz pairs) of a flat grid of (2 half + 1) lines each way, `cell` apart, centred on the origin, y = 0. */
export function gridSegments(half: number, cell: number): Float32Array {
  const n = 2 * half + 1;
  const out = new Float32Array(2 * n * 2 * 3);
  const e = half * cell;
  let k = 0;
  const put = (x: number, z: number) => { out[k++] = x; out[k++] = 0; out[k++] = z; };
  for (let i = -half; i <= half; i++) {
    put(i * cell, -e); put(i * cell, e); // line along Z
    put(-e, i * cell); put(e, i * cell); // line along X
  }
  return out;
}

/** One smoothing step of the camera focus toward the thorax: a fraction `alpha` of the way (1 = jump), settled and
 *  snapped once within `eps`. */
export function followStep(cur: ArrayLike<number>, want: ArrayLike<number>, alpha: number, eps: number): { pos: Vec3; settled: boolean } {
  const d: Vec3 = [want[0] - cur[0], want[1] - cur[1], want[2] - cur[2]];
  if (alpha >= 1 || Math.hypot(d[0], d[1], d[2]) <= eps) return { pos: [want[0], want[1], want[2]], settled: true };
  return { pos: [cur[0] + d[0] * alpha, cur[1] + d[1] * alpha, cur[2] + d[2] * alpha], settled: false };
}

/** Flight wing pose from one frame of the server's `wings` (6 numbers, degrees, flybody convention: left [stroke, deviation,
 *  rotation] then right): wing_*_stroke/dev/rot for the viewer, which composes R_up(stroke) R_fore(dev) R_span(rotation) about
 *  the wing hinge (viewer axes: up = Y, fore = Z, lateral span = X). The flybody model is a different fly and its angle
 *  conventions are not documented in the data, so the signs are OUR choice: the two wings mirror each other, stroke > 0 sweeps
 *  forward, deviation > 0 lifts, rotation > 0 raises the leading edge. Right-hand rule about the viewer axes: the left wing (at +X)
 *  swept forward turns about -Y, the right (at -X) about +Y; deviation +X / -X; rotation -X for both (mirror of a rotation about
 *  the lateral axis keeps its sense). */
export function flightWingPose(w: ArrayLike<number>): Partial<Record<JointId, number>> {
  if (w.length !== 6 || !Array.from(w).every(Number.isFinite)) return {};
  return {
    wing_l_stroke: -w[0], wing_l_dev: w[1], wing_l_rot: -w[2],
    wing_r_stroke: w[3], wing_r_dev: -w[4], wing_r_rot: -w[5],
  };
}
