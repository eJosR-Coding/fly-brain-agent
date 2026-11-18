import { describe, expect, it } from "vitest";
import { escapeAt, clipMatrices, worldPose } from "../src/bodyClip";
import { LIVE_FRAME_MS, parseLiveChunk, parseLiveHeader } from "../src/liveFrames";

import { Q, chunk, header } from "./liveFixtures";

describe("parseLiveHeader", () => {
  it("accepts the server's header", () => {
    const h = parseLiveHeader(header());
    expect(h.bodies).toEqual(["lf_coxa", "lf_tibia"]);
    expect(h.sampleRateHz).toBe(200);
  });
  it.each([
    ["not an object", 3],
    ["no bodies", { ...header(), bodies: [] }],
    ["neutral of the wrong length", { ...header(), neutral: { pos_um: [[0, 0, 0]], quat: [[Q, 0, 0, 0]] } }],
    ["bad quat_scale", { ...header(), quat_scale: 0 }],
    ["bad sample rate", { ...header(), sample_rate_hz: -1 }],
  ])("rejects %s", (_n, raw) => {
    expect(() => parseLiveHeader(raw)).toThrow();
  });
});

describe("parseLiveChunk", () => {
  const h = parseLiveHeader(header());
  it("builds a one-shot clip at the header's sample rate, not looping", () => {
    const { chunk: c } = parseLiveChunk(chunk(), h, null);
    expect(c.clip.nFrames).toBe(10);
    expect(c.clip.loop).toBe(false);
    expect(c.chunkMs).toBe(50);
    expect(c.clip.durationS).toBeCloseTo(0.05, 9);
    expect(LIVE_FRAME_MS).toBe(5);
  });
  it("turns frame positions into leg matrices (fly x -> viewer Z, um -> mm)", () => {
    const { chunk: c } = parseLiveChunk(chunk(), h, null);
    expect(clipMatrices(c.clip, 0).get("lf_coxa")![14]).toBeCloseTo(0, 9);
    expect(clipMatrices(c.clip, 0.005).get("lf_coxa")![14]).toBeCloseTo(0.1, 9); // frame 1
    expect(clipMatrices(c.clip, 99).get("lf_coxa")![14]).toBeCloseTo(0.9, 9); // holds the last frame
  });
  it("makes the thorax world relative to the session: z is relative to the first frame ever seen", () => {
    const a = parseLiveChunk(chunk(), h, null);
    expect(a.z0).toBe(800);
    const w0 = worldPose(a.chunk.clip, 0)!;
    expect(w0.offset).toEqual([0, 0, 0]);
    const w1 = worldPose(a.chunk.clip, 0.005)!;
    expect(w1.offset[2]).toBeCloseTo(1, 9); // 1000 um forward
    const b = parseLiveChunk(chunk(10, { thorax: { pos_um: Array.from({ length: 10 }, () => [10000, 0, 1000]), yaw_deg: Array.from({ length: 10 }, () => 0) } }), h, a.z0);
    expect(b.z0).toBe(800);
    const wb = worldPose(b.chunk.clip, 0)!;
    expect(wb.offset[2]).toBeCloseTo(10, 9);
    expect(wb.offset[1]).toBeCloseTo(0.2, 9); // 200 um above the start height
  });
  it("applies the yaw about the vertical axis (positive = to the fly's left)", () => {
    const c = chunk(10, { thorax: { pos_um: Array.from({ length: 10 }, () => [0, 0, 0]), yaw_deg: Array.from({ length: 10 }, () => 90) } });
    const { chunk: p } = parseLiveChunk(c, h, null);
    const m = worldPose(p.clip, 0)!.matrix;
    expect(m[2]).toBeCloseTo(-1, 9);
    expect(m[8]).toBeCloseTo(1, 9);
  });
  it("carries the escape flag", () => {
    expect(escapeAt(parseLiveChunk(chunk(10, { escape: true }), h, null).chunk.clip, 0.01)).toBe(true);
    expect(escapeAt(parseLiveChunk(chunk(), h, null).chunk.clip, 0.01)).toBe(false);
  });
  it("keeps the readouts the status line needs", () => {
    const { chunk: c } = parseLiveChunk(chunk(10, { speed_label: "x0.20 real time", speed_factor: 0.2, backend: "gpu", fallback: null, body_mode: "recorded gait" }), h, null);
    expect(c.info).toMatchObject({ speedLabel: "x0.20 real time", speedFactor: 0.2, backend: "gpu", fallback: null, bodyMode: "recorded gait" });
  });
  it("tolerates missing optional readouts", () => {
    const { chunk: c } = parseLiveChunk(chunk(), h, null);
    expect(c.info).toMatchObject({ speedLabel: null, backend: null, fallback: null, bodyMode: null });
  });
  it.each([
    ["no frames (binary was not requested)", chunk(10, { frames: null })],
    ["a frame with the wrong body count", chunk(10, { frames: Array.from({ length: 10 }, () => [[0, 0, 0, Q, 0, 0, 0]]) })],
    ["no frames at all", chunk(0)],
    ["a thorax of the wrong length", chunk(10, { thorax: { pos_um: [[0, 0, 0]], yaw_deg: [0] } })],
    ["no thorax", chunk(10, { thorax: undefined })],
    ["a non-finite number", chunk(10, { frames: Array.from({ length: 10 }, () => [[NaN, 0, 0, Q, 0, 0, 0], [0, 0, 0, Q, 0, 0, 0]]) })],
    ["a bad chunk_ms", chunk(10, { chunk_ms: 0 })],
    ["a bad seq", chunk(10, { seq: -1 })],
  ])("rejects %s", (_n, raw) => {
    expect(() => parseLiveChunk(raw, h, null)).toThrow();
  });
});
