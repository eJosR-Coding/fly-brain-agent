import { describe, expect, it } from "vitest";
import { createPlayback } from "../src/livePlayback";

const c = (seq: number, chunkMs = 50) => ({ seq, chunkMs });

describe("createPlayback (jitter buffer on the chunks' own time base)", () => {
  it("shows nothing before the first chunk", () => {
    const p = createPlayback<ReturnType<typeof c>>();
    expect(p.advance(16)).toBeNull();
    expect(p.bufferedMs).toBe(0);
  });
  it("plays at real simulated speed once the buffer holds the target", () => {
    const p = createPlayback<ReturnType<typeof c>>({ targetMs: 100 });
    for (let i = 0; i < 3; i++) p.push(c(i));
    const r = p.advance(20)!;
    expect(r.item.seq).toBe(0);
    expect(r.tSec).toBeCloseTo(0.02, 9);
    expect(r.rate).toBe(1);
  });
  it("slows down in proportion to how little is buffered (never faster than 1x)", () => {
    const p = createPlayback<ReturnType<typeof c>>({ targetMs: 100 });
    p.push(c(0)); // 50 ms buffered of a 100 ms target: half speed
    const r = p.advance(20)!;
    expect(r.rate).toBeCloseTo(0.5, 9);
    expect(r.tSec).toBeCloseTo(0.01, 9);
  });
  it("moves on to the next chunk carrying the leftover time", () => {
    const p = createPlayback<ReturnType<typeof c>>({ targetMs: 100 });
    for (let i = 0; i < 4; i++) p.push(c(i));
    p.advance(45);
    const r = p.advance(10)!; // 45 + 10 = 55 ms: 5 ms into chunk 1
    expect(r.item.seq).toBe(1);
    expect(r.tSec).toBeCloseTo(0.005, 9);
  });
  it("holds the last frame when the buffer runs dry and resumes from the next chunk", () => {
    const p = createPlayback<ReturnType<typeof c>>({ targetMs: 50 });
    p.push(c(0));
    p.advance(60);
    const held = p.advance(16)!;
    expect(held.item.seq).toBe(0);
    expect(held.tSec).toBeCloseTo(0.05, 9);
    expect(held.rate).toBe(0);
    expect(p.bufferedMs).toBe(0);
    p.push(c(1));
    expect(p.advance(10)!.item.seq).toBe(1);
  });
  it("drops the oldest chunks beyond the cap and counts them", () => {
    const p = createPlayback<ReturnType<typeof c>>({ targetMs: 100, maxChunks: 4 });
    for (let i = 0; i < 7; i++) p.push(c(i));
    expect(p.dropped).toBe(3);
    expect(p.advance(1)!.item.seq).toBe(3);
  });
  it("reset empties the buffer and the counters", () => {
    const p = createPlayback<ReturnType<typeof c>>({ maxChunks: 2 });
    for (let i = 0; i < 4; i++) p.push(c(i));
    p.reset();
    expect(p.dropped).toBe(0);
    expect(p.bufferedMs).toBe(0);
    expect(p.advance(16)).toBeNull();
  });
  it("ignores a non-positive or non-finite step", () => {
    const p = createPlayback<ReturnType<typeof c>>();
    p.push(c(0)); p.push(c(1));
    const a = p.advance(10)!.tSec;
    expect(p.advance(0)!.tSec).toBe(a);
    expect(p.advance(-5)!.tSec).toBe(a);
    expect(p.advance(NaN)!.tSec).toBe(a);
  });
  it("follows a slow producer smoothly: x0.2 supply plays at about x0.2 without long stalls", () => {
    const p = createPlayback<ReturnType<typeof c>>({ targetMs: 100 });
    let seq = 0, sinceChunk = 250, played = 0, stalled = 0;
    const frames = 1000; // 16 s of 16 ms frames
    for (let f = 0; f < frames; f++) {
      sinceChunk += 16;
      while (sinceChunk >= 250) { sinceChunk -= 250; p.push(c(seq++)); } // 50 ms of motion every 250 ms of wall time
      const r = p.advance(16);
      if (!r) continue;
      played += r.rate * 16;
      if (r.rate === 0) stalled++;
    }
    expect(played / (frames * 16)).toBeGreaterThan(0.15);
    expect(played / (frames * 16)).toBeLessThan(0.25);
    expect(stalled / frames).toBeLessThan(0.1);
  });
});
