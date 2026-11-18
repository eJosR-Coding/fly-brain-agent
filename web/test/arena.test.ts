import { describe, expect, it } from "vitest";
import { SANDBOX_INTERIOR_MM, arenaBounds, arenaToViewer, parseArena, cubeBounds } from "../src/bodyCube";
import { createSandbox } from "../src/bodySandbox";
import { parseLiveChunk, parseLiveHeader } from "../src/liveFrames";
import { createLiveSession, liveCaption, type SocketLike } from "../src/liveSession";
import { worldPose } from "../src/bodyClip";
import { chunk, header } from "./liveFixtures";

const theme = { highlightHex: "#ffffff" } as never;
const serverArena = (over: Record<string, unknown> = {}) => ({
  kind: "glass_cube", interior_mm: 50, bounds_mm: [-25, 25, -25, 25], note: "world rule: glass walls stop the fly; not a brain decision", ...over,
});

describe("parseArena", () => {
  it("reads the server's arena", () => {
    expect(parseArena(serverArena())).toEqual({ kind: "glass_cube", interiorMm: 50, boundsMm: [-25, 25, -25, 25], note: expect.stringContaining("world rule") });
  });
  it.each([null, undefined, 5, "x", [], {}, serverArena({ kind: "sphere" }), serverArena({ interior_mm: -1 }), serverArena({ interior_mm: "50" }),
    serverArena({ bounds_mm: [1, 0, -1, 1] }), serverArena({ bounds_mm: [0, 1, 2] }), serverArena({ bounds_mm: [0, 1, 0, NaN] }),
  ])("rejects %j and falls back to the constant", (raw) => {
    expect(parseArena(raw)).toBeNull();
  });
});

describe("arena geometry", () => {
  it("the cube is drawn from the bounds: fly x -> viewer Z, fly y -> viewer X, the floor at floorY", () => {
    const a = parseArena(serverArena({ bounds_mm: [-10, 30, -5, 15], interior_mm: 40 }))!;
    const b = arenaBounds(-0.4, a);
    expect(b.min).toEqual([-5, -0.4, -10]);
    expect(b.max).toEqual([15, -0.4 + 40, 30]);
  });
  it("the default arena equals the fallback cube", () => {
    const b = arenaBounds(-0.4, parseArena(serverArena())!);
    expect(b).toEqual(cubeBounds(-0.4, SANDBOX_INTERIOR_MM));
  });
  it("the drawn wall coincides with the thorax world pose of a fly standing on the wall (same axis mapping)", () => {
    const a = parseArena(serverArena({ bounds_mm: [-10, 30, -5, 15] }))!;
    const b = arenaBounds(0, a);
    const h = parseLiveHeader(header());
    // root exactly at the forward wall x = 30 mm and the left wall y = 15 mm (server um, relative to the start)
    const m = chunk(2, { thorax: { pos_um: [[30000, 15000, 800], [30000, 15000, 800]], yaw_deg: [0, 0] } });
    const { chunk: c } = parseLiveChunk(m, h, null);
    const w = worldPose(c.clip, 0)!;
    expect(w.offset[2]).toBeCloseTo(b.max[2], 9); // viewer Z = forward wall
    expect(w.offset[0]).toBeCloseTo(b.max[0], 9); // viewer X = left wall
    expect(arenaToViewer(30, 15)).toEqual([b.max[0], b.max[2]]);
  });
  it("createSandbox takes the arena bounds", () => {
    const a = parseArena(serverArena({ bounds_mm: [-10, 30, -5, 15], interior_mm: 40 }))!;
    const s = createSandbox(theme, -0.4, SANDBOX_INTERIOR_MM, a);
    expect(s.bounds).toEqual(arenaBounds(-0.4, a));
    const pos = (s.group.getObjectByName("cube-edges") as unknown as { geometry: { getAttribute(n: string): { array: Float32Array } } }).geometry.getAttribute("position").array;
    expect(Math.max(...Array.from(pos).filter((_, i) => i % 3 === 2))).toBeCloseTo(30);
  });
});

class FakeSocket implements SocketLike {
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: ((ev: { code: number; reason: string }) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;
  send() {}
  close() {}
  emit(msg: unknown) { this.onmessage?.({ data: JSON.stringify(msg) }); }
}

function harness() {
  const sock = new FakeSocket();
  const arenas: unknown[] = [];
  const captions: string[] = [];
  const s = createLiveSession({
    url: "ws://x", openSocket: () => sock,
    sinks: { setFrame() {}, setWorld() {}, setPose() {}, setCaption: (t) => captions.push(t), setArena: (a) => arenas.push(a) },
    every: () => () => {}, defaultCaption: "d",
  });
  s.start({ preset: null, rateHz: 0 });
  sock.onopen?.();
  const started = (extra: Record<string, unknown> = {}) => sock.emit({ type: "started", chunk_ms: 50, backend: "cpu", fallback: null, header: header(), caption: "c", note: "n", ...extra });
  return { s, sock, arenas, captions, started };
}

describe("live session arena and wall note", () => {
  it("hands the server's arena to the sink on started", () => {
    const h = harness();
    h.started({ arena: serverArena({ bounds_mm: [-5, 5, -5, 5], interior_mm: 10 }) });
    expect(h.arenas).toEqual([expect.objectContaining({ interiorMm: 10, boundsMm: [-5, 5, -5, 5] })]);
  });
  it("sends null (fallback constant) when the server sends no or a bad arena", () => {
    const a = harness(); a.started(); expect(a.arenas).toEqual([null]);
    const b = harness(); b.started({ arena: { kind: "x" } }); expect(b.arenas).toEqual([null]);
  });
  it("shows an honest caption note while the received chunk is blocked, and drops it when free", () => {
    const h = harness();
    h.started({ arena: serverArena() });
    h.sock.emit(chunk(10, { seq: 0, world: { bounds_mm: [-25, 25, -25, 25], blocked: true, walls: ["x_max"], note: "n" } }));
    expect(h.captions.at(-1)).toContain("stopped by the glass (world rule)");
    h.sock.emit(chunk(10, { seq: 1, world: { bounds_mm: [-25, 25, -25, 25], blocked: false, walls: [], note: "n" } }));
    expect(h.captions.at(-1)).not.toContain("stopped by the glass");
  });
  it("liveCaption appends the note only when blocked", () => {
    const base = { speedLabel: "x1.00 real time", backend: "cpu", fallback: null, bodyMode: "kinematic", dropped: 0, serverCaption: "" };
    expect(liveCaption({ ...base, worldBlocked: true })).toContain("stopped by the glass (world rule)");
    expect(liveCaption({ ...base, worldBlocked: false })).not.toContain("glass");
    expect(liveCaption(base)).not.toContain("glass");
  });
});
