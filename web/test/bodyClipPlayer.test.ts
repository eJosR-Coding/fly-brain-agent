import { describe, expect, it, vi } from "vitest";
import { parseWalkClip } from "../src/bodyClip";
import { createClipPlayer } from "../src/bodyClipPlayer";

const Q = 10000;
/** 2 Hz, 4 frames (2 s), no looping; coxa slides 0..300 um; thorax walks 1 mm forward per frame; escape on the last frame only */
function clip(extra: Record<string, unknown> = {}) {
  return parseWalkClip({
    format: "fly-walk/1", quat_scale: Q, thorax_origin_um: [0, 0, 0], bodies: ["lf_coxa"], sample_rate_hz: 2, n_frames: 4, loop: false,
    neutral: { pos_um: [[0, 0, 0]], quat: [[Q, 0, 0, 0]] },
    frames: [0, 100, 200, 300].map((x) => [[x, 0, 0, Q, 0, 0, 0]]),
    escape: [0, 0, 0, 1],
    thorax_world: { pos_um: [[0, 0, 0], [1000, 0, 0], [2000, 0, 0], [3000, 0, 0]], yaw_deg: [0, 0, 0, 0] },
    ...extra,
  });
}
function harness(reducedMotion = false) {
  const log = { frames: [] as Array<ReadonlyMap<string, ArrayLike<number>> | null>, worlds: [] as Array<unknown>, poses: [] as unknown[] };
  const p = createClipPlayer({
    setFrame: (m) => log.frames.push(m), setWorld: (w) => log.worlds.push(w), setPose: (a) => log.poses.push(a), reducedMotion,
  });
  return { p, log };
}
const z = (m: ReadonlyMap<string, ArrayLike<number>> | null) => (m ? m.get("lf_coxa")![14] : null);

describe("createClipPlayer (clip driven by the recording's clock)", () => {
  it("does nothing without a clip", () => {
    const { p, log } = harness();
    p.onTime(500);
    expect(log.frames).toEqual([]);
    expect(p.active).toBe(false);
  });
  it("shows the frame, root transform and wing pose for the playback time", () => {
    const { p, log } = harness();
    p.setClip(clip());
    expect(p.active).toBe(true);
    p.onTime(500); // 0.5 s = frame 1
    expect(z(log.frames.at(-1)!)).toBeCloseTo(0.1, 9);
    expect((log.worlds.at(-1) as { offset: number[] }).offset[2]).toBeCloseTo(1, 9);
    expect(log.poses.at(-1)).toEqual({});
    p.onTime(1500); // frame 3: escape
    expect(log.poses.at(-1)).toEqual({ wing_l: 70, wing_r: 70 });
  });
  it("holds the last frame once the recording is past the clip, and does not repeat unchanged poses", () => {
    const { p, log } = harness();
    p.setClip(clip());
    p.onTime(1500);
    const n = log.poses.length;
    p.onTime(9000);
    expect(z(log.frames.at(-1)!)).toBeCloseTo(0.3, 9);
    expect(log.poses.length).toBe(n);
  });
  it("null time (nothing loaded / playhead hidden) returns the body to rest", () => {
    const { p, log } = harness();
    p.setClip(clip());
    p.onTime(500);
    p.onTime(null);
    expect(log.frames.at(-1)).toBeNull();
    expect(log.worlds.at(-1)).toBeNull();
    expect(log.poses.at(-1)).toEqual({});
  });
  it("clear() rests the body and ignores later ticks", () => {
    const { p, log } = harness();
    p.setClip(clip());
    p.onTime(500);
    p.clear();
    expect(p.active).toBe(false);
    const n = log.frames.length;
    p.onTime(700);
    expect(log.frames.length).toBe(n);
    expect(log.frames.at(-1)).toBeNull();
  });
  it("clear() with no clip is quiet", () => {
    const { p, log } = harness();
    p.clear();
    expect(log.frames).toEqual([]);
  });
  it("a clip without thorax_world never moves the world", () => {
    const { p, log } = harness();
    p.setClip(clip({ thorax_world: undefined }));
    p.onTime(500);
    expect(log.worlds.at(-1)).toBeNull();
  });
  it("reduced motion shows the final frame at once and ignores the clock", () => {
    const { p, log } = harness(true);
    p.setClip(clip());
    expect(z(log.frames.at(-1)!)).toBeCloseTo(0.3, 9);
    expect(log.poses.at(-1)).toEqual({ wing_l: 70, wing_r: 70 });
    const n = log.frames.length;
    p.onTime(0);
    p.onTime(null);
    expect(log.frames.length).toBe(n);
  });
  it("a sink that throws does not break the clock (the next tick tries again)", () => {
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    let boom = true;
    const p = createClipPlayer({ setFrame: () => { if (boom) throw new Error("x"); }, setWorld() {}, setPose() {} });
    p.setClip(clip());
    expect(() => p.onTime(100)).not.toThrow();
    boom = false;
    expect(() => p.onTime(200)).not.toThrow();
    err.mockRestore();
  });
});
