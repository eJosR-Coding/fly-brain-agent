import { describe, expect, it } from "vitest";
import { clipMatrices, escapeAt, parseWalkClip, worldPose } from "../src/bodyClip";
import { applyMatrix as apply } from "../src/bodyRig";
const applyMatrix = (m: ArrayLike<number>, p: ArrayLike<number>) => apply(p, m);

const Q = 10000;
const REST = [0, 0, 0, Q, 0, 0, 0];

/** 2 Hz, n frames, one body at the neutral pose; optional loop / escape / thorax_world fields. */
function raw(n: number, extra: Record<string, unknown> = {}) {
  const frames = Array.from({ length: n }, (_, f) => [[f * 100, 0, 0, Q, 0, 0, 0]]); // lf_coxa slides along x
  return {
    format: "fly-walk/1", quat_scale: Q, thorax_origin_um: [0, 0, 0], bodies: ["lf_coxa"], sample_rate_hz: 2, n_frames: n,
    neutral: { pos_um: [[0, 0, 0]], quat: [[Q, 0, 0, 0]] }, frames, ...extra,
  };
}
const world = (pos: number[][], yaw: number[]) => ({ pos_um: pos, yaw_deg: yaw });
const zOf = (t: number, c: ReturnType<typeof parseWalkClip>) => clipMatrices(c, t).get("lf_coxa")![14];

describe("loop flag", () => {
  it("loops by default and when loop is true", () => {
    for (const c of [parseWalkClip(raw(4)), parseWalkClip(raw(4, { loop: true }))]) {
      expect(c.loop).toBe(true);
      expect(zOf(2.0, c)).toBeCloseTo(0, 9); // t = duration wraps to frame 0
    }
  });
  it("loop:false holds the last frame past the end and before the start", () => {
    const c = parseWalkClip(raw(4, { loop: false }));
    expect(c.loop).toBe(false);
    expect(zOf(1.5, c)).toBeCloseTo(0.3, 9); // frame 3 at 1.5 s
    expect(zOf(2.0, c)).toBeCloseTo(0.3, 9);
    expect(zOf(99, c)).toBeCloseTo(0.3, 9);
    expect(zOf(-5, c)).toBeCloseTo(0, 9);
  });
  it("loop:false still interpolates inside the clip, and a non-boolean loop is ignored", () => {
    const c = parseWalkClip(raw(4, { loop: false }));
    expect(zOf(0.25, c)).toBeCloseTo(0.05, 9);
    expect(parseWalkClip(raw(4, { loop: "no" })).loop).toBe(true);
  });
});

describe("escape flags", () => {
  it("are absent -> never escape", () => {
    expect(escapeAt(parseWalkClip(raw(4)), 0.5)).toBe(false);
  });
  it("read the flag of the frame at t (held after the end when not looping)", () => {
    const c = parseWalkClip(raw(4, { loop: false, escape: [0, 0, 1, 1] }));
    expect(escapeAt(c, 0.0)).toBe(false);
    expect(escapeAt(c, 0.5)).toBe(false); // frame 1
    expect(escapeAt(c, 1.0)).toBe(true); // frame 2
    expect(escapeAt(c, 50)).toBe(true);
  });
  it("a malformed escape list is dropped, not fatal", () => {
    expect(escapeAt(parseWalkClip(raw(4, { escape: [1, 1] })), 0)).toBe(false);
  });
});

describe("thorax_world -> root transform", () => {
  it("is null without thorax_world", () => {
    expect(worldPose(parseWalkClip(raw(2)), 0)).toBeNull();
  });
  it("maps fly x forward / y left / z up to viewer (Z, X, Y) in mm, relative to the start", () => {
    const c = parseWalkClip(raw(2, { loop: false, thorax_world: world([[0, 0, 800], [1000, 2000, 850]], [0, 0]) }));
    const w0 = worldPose(c, 0)!;
    expect(w0.offset).toEqual([0, 0, 0]);
    const w1 = worldPose(c, 0.5)!;
    // forward 1 mm -> viewer Z, left 2 mm -> viewer X, up 0.05 mm -> viewer Y
    expect(w1.offset[0]).toBeCloseTo(2, 9);
    expect(w1.offset[1]).toBeCloseTo(0.05, 9);
    expect(w1.offset[2]).toBeCloseTo(1, 9);
    expect(Array.from(applyMatrix(w1.matrix, [0, 0, 0]))).toEqual([2, 0.05, 1]);
  });
  it("yaw is a rotation about the vertical (viewer Y) axis through the thorax origin; positive = left", () => {
    const c = parseWalkClip(raw(2, { loop: false, thorax_origin_um: [0, 0, 0], thorax_world: world([[0, 0, 0], [0, 0, 0]], [0, 90]) }));
    const m = worldPose(c, 0.5)!.matrix;
    const fwd = applyMatrix(m, [0, 0, 1]); // fly forward (viewer +Z) turned 90 deg left -> viewer +X (fly left)
    expect(fwd[0]).toBeCloseTo(1, 9);
    expect(fwd[1]).toBeCloseTo(0, 9);
    expect(fwd[2]).toBeCloseTo(0, 9);
    const up = applyMatrix(m, [0, 1, 0]);
    expect(up[1]).toBeCloseTo(1, 9);
  });
  it("rotates about the thorax origin, not the mesh origin", () => {
    // thorax at fly (1000, 0, 0) um = viewer (0, 0, 1) mm
    const c = parseWalkClip(raw(2, { loop: false, thorax_origin_um: [1000, 0, 0], thorax_world: world([[0, 0, 0], [0, 0, 0]], [0, 180]) }));
    const m = worldPose(c, 0.5)!.matrix;
    const p = applyMatrix(m, [0, 0, 1]);
    expect(p[0]).toBeCloseTo(0, 9);
    expect(p[2]).toBeCloseTo(1, 9); // the thorax itself does not move
    const q = applyMatrix(m, [0, 0, 2]);
    expect(q[2]).toBeCloseTo(0, 9); // a point 1 mm ahead of it ends 1 mm behind
  });
  it("interpolates between frames, unwrapped yaw included, and holds the last frame", () => {
    const c = parseWalkClip(raw(3, { loop: false, thorax_world: world([[0, 0, 0], [1000, 0, 0], [2000, 0, 0]], [0, 100, 200]) }));
    expect(worldPose(c, 0.25)!.offset[2]).toBeCloseTo(0.5, 9);
    expect(worldPose(c, 9)!.offset[2]).toBeCloseTo(2, 9);
    const mid = applyMatrix(worldPose(c, 0.75)!.matrix, [0, 0, 1 + 1.5]);
    expect(Number.isFinite(mid[0])).toBe(true);
  });
  it("a malformed thorax_world is dropped", () => {
    expect(worldPose(parseWalkClip(raw(2, { thorax_world: world([[0, 0, 0]], [0, 0]) })), 0)).toBeNull();
    expect(worldPose(parseWalkClip(raw(2, { thorax_world: world([[0, 0, 0], [0, 0, "x"]], [0, 0]) })), 0)).toBeNull();
  });
});
