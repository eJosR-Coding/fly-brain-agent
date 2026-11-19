import { describe, expect, it } from "vitest";
import { MAPPING } from "../src/bodyPose";
import { escapePose, flightWingPose, followStep, gridSegments, gridSnap, WING_RAISE_DEG } from "../src/bodyWorld";

describe("escape -> wings", () => {
  it("raises both wings by the dnp01 mapping's full angle, and nothing otherwise", () => {
    expect(WING_RAISE_DEG).toBe(MAPPING.dnp01.maxDeg);
    expect(escapePose(true)).toEqual({ wing_l: WING_RAISE_DEG, wing_r: WING_RAISE_DEG });
    expect(escapePose(false)).toEqual({});
  });
});

describe("floor grid", () => {
  it("snaps to whole cells so the pattern stays fixed in the world while the fly moves", () => {
    expect(gridSnap([0.26, 5, 0.9], 0.5)).toEqual([0, 0.5]);
    expect(gridSnap([-0.01, 0, -0.51], 0.5)).toEqual([-0.5, -1]);
    expect(gridSnap([1, 0, 1], 0.5)).toEqual([1, 1]);
  });
  it("draws (2 half + 1) lines each way, two vertices per line, centred on the origin", () => {
    const seg = gridSegments(2, 0.5);
    expect(seg.length).toBe(2 * 5 * 2 * 3); // 2 directions x 5 lines x 2 vertices x xyz
    const xs = Array.from({ length: seg.length / 3 }, (_, k) => seg[k * 3]);
    const zs = Array.from({ length: seg.length / 3 }, (_, k) => seg[k * 3 + 2]);
    expect(Math.max(...xs)).toBeCloseTo(1, 6);
    expect(Math.min(...xs)).toBeCloseTo(-1, 6);
    expect(Math.max(...zs)).toBeCloseTo(1, 6);
    for (let k = 1; k < seg.length; k += 3) expect(seg[k]).toBe(0); // flat
  });
});

describe("camera follow", () => {
  it("moves a fraction of the way and reports not settled", () => {
    const r = followStep([0, 0, 0], [10, 0, 0], 0.25, 1e-3);
    expect(r.pos).toEqual([2.5, 0, 0]);
    expect(r.settled).toBe(false);
  });
  it("snaps and settles inside eps, and alpha 1 jumps straight there (reduced motion)", () => {
    expect(followStep([9.9995, 0, 0], [10, 0, 0], 0.25, 1e-3)).toEqual({ pos: [10, 0, 0], settled: true });
    expect(followStep([0, 0, 0], [3, 4, 5], 1, 1e-3)).toEqual({ pos: [3, 4, 5], settled: true });
  });
  it("converges", () => {
    let p: [number, number, number] = [0, 0, 0];
    for (let k = 0; k < 200; k++) p = followStep(p, [1, 2, 3], 0.2, 1e-4).pos as [number, number, number];
    expect(p).toEqual([1, 2, 3]);
  });
});

describe("flightWingPose", () => {
  it("mirrors the two wings: stroke and deviation flip sign, rotation keeps its sense", () => {
    const p = flightWingPose([10, 20, 30, 10, 20, 30]);
    expect(p.wing_l_stroke).toBe(-p.wing_r_stroke!);
    expect(p.wing_l_dev).toBe(-p.wing_r_dev!);
    expect(p.wing_l_rot).toBe(p.wing_r_rot);
  });
  it("is empty for anything that is not six finite numbers", () => {
    expect(flightWingPose([1, 2, 3])).toEqual({});
    expect(flightWingPose([1, 2, 3, 4, 5, NaN])).toEqual({});
  });
});
