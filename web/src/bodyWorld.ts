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
