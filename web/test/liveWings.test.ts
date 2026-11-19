import { describe, expect, it } from "vitest";
import { createLiveSession, type SocketLike } from "../src/liveSession";
import { chunk, header } from "./liveFixtures";

class Sock implements SocketLike {
  onopen: (() => void) | null = null; onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: ((ev: { code: number; reason: string }) => void) | null = null; onerror: ((ev: unknown) => void) | null = null;
  send() {} close() {}
}

describe("airborne chunk -> wing poses", () => {
  it("setPose receives a different wing pose on successive display frames", () => {
    const poses: Array<Record<string, number>> = [];
    const sock = new Sock();
    const s = createLiveSession({
      url: "ws://x", openSocket: () => sock,
      sinks: { setFrame() {}, setWorld() {}, setPose: (a) => poses.push(a as Record<string, number>), setCaption() {} },
      every: () => () => {}, defaultCaption: "d",
    });
    s.start({ preset: null, rateHz: 0 });
    sock.onopen?.();
    sock.onmessage?.({ data: JSON.stringify({ type: "started", chunk_ms: 50, backend: "gpu", fallback: null, header: header(), caption: "c", note: "n", binary_frames: false }) });
    const wings = Array.from({ length: 10 }, (_, i) => [-40 + 12 * i, 0, 10 * i, -40 + 12 * i, 0, 10 * i]);
    const cycle = Array.from({ length: 16 }, (_, i) => [-50 + 8 * i, 0, 5 * i, -50 + 8 * i, 0, 5 * i]);
    for (let q = 0; q < 3; q++) sock.onmessage?.({ data: JSON.stringify(chunk(10, { seq: q, wings, wings_note: "n", wing_cycle: cycle })) });
    for (let i = 0; i < 40; i++) s.tick(i * 16.7); // 60 fps; even with the playhead held on one chunk
    const strokes = new Set(poses.map((p) => p.wing_l_stroke));
    expect(strokes.size).toBeGreaterThan(5); // flaps on the display clock, not simulated time;
  });
});
