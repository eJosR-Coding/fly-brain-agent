import { describe, expect, it, vi } from "vitest";
import { createLiveSession, closeMessage, liveCaption, type SocketLike } from "../src/liveSession";
import { chunk, header } from "./liveFixtures";

class FakeSocket implements SocketLike {
  sent: Array<Record<string, unknown>> = [];
  closed: Array<{ code?: number; reason?: string }> = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: ((ev: { code: number; reason: string }) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;
  send(data: string) { this.sent.push(JSON.parse(data)); }
  close(code?: number, reason?: string) { this.closed.push({ code, reason }); }
  open() { this.onopen?.(); }
  emit(msg: unknown) { this.onmessage?.({ data: typeof msg === "string" ? msg : JSON.stringify(msg) }); }
  drop(code: number, reason = "") { this.onclose?.({ code, reason }); }
}

function harness() {
  const sockets: FakeSocket[] = [];
  const log = { frames: [] as unknown[], worlds: [] as unknown[], poses: [] as unknown[], captions: [] as string[] };
  const timers: Array<{ fn: () => void; ms: number; cancelled: boolean }> = [];
  const s = createLiveSession({
    url: "ws://x/api/ws/live",
    openSocket: () => { const k = new FakeSocket(); sockets.push(k); return k; },
    sinks: {
      setFrame: (m) => log.frames.push(m), setWorld: (w) => log.worlds.push(w), setPose: (a) => log.poses.push(a),
      setCaption: (t) => log.captions.push(t),
    },
    every: (fn, ms) => { const t = { fn, ms, cancelled: false }; timers.push(t); return () => { t.cancelled = true; }; },
    defaultCaption: "default caption",
  });
  const sock = () => sockets.at(-1)!;
  /** open + started */
  const live = () => {
    expect(s.start({ preset: "dng100", rateHz: 50 })).toBe(true);
    sock().open();
    sock().emit({ type: "started", chunk_ms: 50, backend: "gpu", fallback: null, header: header(), caption: "Body drive: our mapping", note: "n", binary_frames: false });
  };
  return { s, sockets, sock, log, timers, live };
}

describe("createLiveSession: connect and start", () => {
  it("starts idle and sends the start command once the socket opens", () => {
    const { s, sock } = harness();
    expect(s.status.state).toBe("idle");
    expect(s.start({ preset: "dng100", rateHz: 50 })).toBe(true);
    expect(s.status.state).toBe("connecting");
    expect(sock().sent).toEqual([]);
    sock().open();
    expect(s.status.state).toBe("starting");
    expect(sock().sent).toEqual([{ type: "start", preset: "dng100", rate_hz: 50 }]);
  });
  it("includes optional start fields only when given", () => {
    const { s, sock } = harness();
    s.start({ preset: null, rateHz: 0, seed: 3, chunkMs: 100 });
    sock().open();
    expect(sock().sent[0]).toEqual({ type: "start", preset: null, rate_hz: 0, seed: 3, chunk_ms: 100 });
  });
  it("becomes live on `started` and takes the server's backend, caption and note", () => {
    const { s, live } = harness();
    live();
    expect(s.status).toMatchObject({ state: "live", backend: "gpu", fallback: null, serverCaption: "Body drive: our mapping", note: "n" });
  });
  it("refuses a second start while a session is in flight", () => {
    const { s, sockets, live } = harness();
    live();
    expect(s.start({ preset: null, rateHz: 0 })).toBe(false);
    expect(sockets).toHaveLength(1);
  });
  it("rejects an unreadable header with a clear message and closes", () => {
    const { s, sock } = harness();
    s.start({ preset: null, rateHz: 0 });
    sock().open();
    sock().emit({ type: "started", header: { bodies: [] } });
    expect(s.status.state).toBe("closed");
    expect(s.status.message).toMatch(/header/);
    expect(sock().closed).toHaveLength(1);
  });
  it("turns a socket that cannot be created into a closed state", () => {
    const s = createLiveSession({
      url: "x", openSocket: () => { throw new Error("bad url"); }, every: () => () => {}, defaultCaption: "d",
      sinks: { setFrame() {}, setWorld() {}, setPose() {}, setCaption() {} },
    });
    expect(s.start({ preset: null, rateHz: 0 })).toBe(false);
    expect(s.status).toMatchObject({ state: "closed" });
    expect(s.status.message).toMatch(/bad url/);
  });
});

describe("createLiveSession: chunks and playback", () => {
  it("applies leg matrices and the thorax pose on tick, at the chunk's own time base", () => {
    const { s, sock, log, live } = harness();
    live();
    for (let i = 0; i < 3; i++) sock().emit(chunk(10, { seq: i, t_ms: 50 * (i + 1) }));
    expect(s.tick(1000)).toBe(true); // the first tick only sets the clock
    expect(s.tick(1020)).toBe(true); // 20 ms later: frame 4 of chunk 0 (buffer is full: x1)
    const mats = log.frames.at(-1) as Map<string, ArrayLike<number>>;
    expect(mats.get("lf_coxa")![14]).toBeCloseTo(0.4, 9);
    expect((log.worlds.at(-1) as { offset: number[] }).offset[2]).toBeCloseTo(4, 9);
  });
  it("clamps a long gap between draws (a sleeping tab) instead of jumping ahead", () => {
    const { s, sock, log, live } = harness();
    live();
    for (let i = 0; i < 3; i++) sock().emit(chunk(10, { seq: i }));
    s.tick(0);
    s.tick(60_000);
    const mats = log.frames.at(-1) as Map<string, ArrayLike<number>>;
    expect(mats.get("lf_coxa")![14]).toBeLessThanOrEqual(0.9 + 1e-9);
    expect(s.status.chunksPlayed).toBeLessThan(3);
  });
  it("does nothing on tick before the first chunk and when not live", () => {
    const { s, log, live } = harness();
    expect(s.tick(10)).toBe(false);
    live();
    expect(s.tick(20)).toBe(true);
    expect(log.frames).toEqual([]);
  });
  it("sets the escape wing pose once per change", () => {
    const { s, sock, log, live } = harness();
    live();
    sock().emit(chunk(10, { seq: 0, escape: true }));
    sock().emit(chunk(10, { seq: 1, escape: true }));
    s.tick(0); s.tick(10); s.tick(20);
    expect(log.poses.filter((p) => Object.keys(p as object).length > 0)).toHaveLength(1);
  });
  it("ignores duplicate and out-of-order chunks", () => {
    const { s, sock, live } = harness();
    live();
    sock().emit(chunk(10, { seq: 1 }));
    sock().emit(chunk(10, { seq: 1 }));
    sock().emit(chunk(10, { seq: 0 }));
    expect(s.status.chunksReceived).toBe(1);
    expect(s.status.ignored).toBe(2);
  });
  it("counts malformed messages and keeps going", () => {
    const { s, sock, live } = harness();
    live();
    sock().emit("not json {");
    sock().emit("[1,2]");
    sock().emit({ type: "chunk", seq: 0 });
    sock().emit({ type: "mystery" });
    sock().emit({ data: 1 });
    expect(s.status.state).toBe("live");
    expect(s.status.malformed).toBe(5);
    sock().emit(chunk(10, { seq: 0 }));
    expect(s.status.chunksReceived).toBe(1);
  });
  it("ignores binary messages (binary frames are never requested)", () => {
    const { s, sock, live } = harness();
    live();
    sock().onmessage?.({ data: new ArrayBuffer(8) });
    expect(s.status.malformed).toBe(1);
  });
  it("shows the server's speed label, backend and fallback honestly in the caption", () => {
    const { sock, log, live } = harness();
    live();
    sock().emit(chunk(10, { seq: 0, speed_label: "x0.20 real time", backend: "cpu", fallback: "GPU too hot (82 C)" }));
    const text = log.captions.at(-1)!;
    expect(text).toContain("x0.20 real time");
    expect(text).toContain("brain: cpu");
    expect(text).toContain("GPU too hot");
    expect(text).toContain("Body drive: our mapping");
  });
  it("shows the body mode when the server sends one", () => {
    const { sock, log, live } = harness();
    live();
    sock().emit(chunk(10, { seq: 0, speed_label: "x1.00 real time", backend: "gpu", body_mode: "recorded physics gait, blended live" }));
    expect(log.captions.at(-1)).toContain("recorded physics gait, blended live");
  });
  it("reports a server error message without ending the session", () => {
    const { s, sock, live } = harness();
    live();
    sock().emit({ type: "error", detail: "preset must be a name or null", command: "stimulus" });
    expect(s.status.state).toBe("live");
    expect(s.status.message).toContain("preset must be a name or null");
  });
});

describe("createLiveSession: commands", () => {
  it("sends stimulus changes only while live", () => {
    const { s, sock, live } = harness();
    expect(s.stimulus("dng100", 50)).toBe(false);
    live();
    expect(s.stimulus("dna02_left", 200)).toBe(true);
    expect(s.stimulus(null, 0)).toBe(true);
    expect(sock().sent.slice(1)).toEqual([
      { type: "stimulus", preset: "dna02_left", rate_hz: 200 },
      { type: "stimulus", preset: null, rate_hz: 0 },
    ]);
  });
  it("pings every ~60 s while live and stops pinging afterwards", () => {
    const { sock, timers, live, s } = harness();
    live();
    expect(timers).toHaveLength(1);
    expect(timers[0].ms).toBe(60_000);
    timers[0].fn();
    expect(sock().sent.at(-1)).toEqual({ type: "ping" });
    s.stop();
    expect(timers[0].cancelled).toBe(true);
  });
  it("stop sends stop, closes, returns the body to rest and goes idle", () => {
    const { s, sock, log, live } = harness();
    live();
    sock().emit(chunk(10, { seq: 0 }));
    s.tick(0); s.tick(10);
    s.stop();
    expect(sock().sent.at(-1)).toEqual({ type: "stop" });
    expect(sock().closed).toHaveLength(1);
    expect(s.status.state).toBe("idle");
    expect(log.frames.at(-1)).toBeNull();
    expect(log.worlds.at(-1)).toBeNull();
    expect(log.poses.at(-1)).toEqual({});
    expect(log.captions.at(-1)).toBe("default caption");
  });
  it("stop while connecting just closes the socket", () => {
    const { s, sock } = harness();
    s.start({ preset: null, rateHz: 0 });
    s.stop();
    expect(sock().sent).toEqual([]);
    expect(sock().closed).toHaveLength(1);
    expect(s.status.state).toBe("idle");
  });
  it("is quiet to stop when idle", () => {
    const { s, log } = harness();
    s.stop();
    expect(log.frames).toEqual([]);
  });
  it("ignores events of a socket that was stopped", () => {
    const { s, sock, live } = harness();
    live();
    const old = sock();
    s.stop();
    old.emit(chunk(10, { seq: 0 }));
    old.drop(1006);
    expect(s.status.state).toBe("idle");
    expect(s.status.chunksReceived).toBe(0);
  });
});

describe("createLiveSession: closing and reconnecting", () => {
  it.each([
    [4409, /another live session/],
    [4503, /uv sync --extra body/],
    [4408, /idle/],
    [4403, /origin/],
    [4400, /bad thing/],
    [1011, /failed/],
    [1006, /lost/],
  ])("explains close code %i", (code, re) => {
    const { s, sock, live } = harness();
    live();
    sock().drop(code, "bad thing");
    expect(s.status.state).toBe("closed");
    expect(s.status.message).toMatch(re);
  });
  it("rests the body and shows the message in the caption when the server closes", () => {
    const { s, sock, log, live } = harness();
    live();
    sock().drop(4409, "busy");
    expect(log.frames.at(-1)).toBeNull();
    expect(log.worlds.at(-1)).toBeNull();
    expect(log.captions.at(-1)).toBe(s.status.message);
  });
  it("explains a connection that never opened", () => {
    const { s, sock } = harness();
    s.start({ preset: null, rateHz: 0 });
    sock().drop(1006);
    expect(s.status.message).toMatch(/could not connect/);
  });
  it("a normal close after `stopped` is idle, not an error", () => {
    const { s, sock, live } = harness();
    live();
    sock().emit({ type: "stopped", reason: "client" });
    sock().drop(1000, "stopped");
    expect(s.status.state).toBe("idle");
  });
  it("never reconnects by itself, only on a new start()", () => {
    const { s, sockets, live, sock } = harness();
    live();
    sock().drop(1006);
    expect(sockets).toHaveLength(1);
    expect(s.start({ preset: "dng100", rateHz: 50 })).toBe(true);
    expect(sockets).toHaveLength(2);
    expect(s.status.state).toBe("connecting");
  });
  it("a new session starts from a clean buffer and the world origin of its own", () => {
    const { s, sock, sockets, log, live } = harness();
    live();
    sock().emit(chunk(10, { seq: 0 }));
    sock().drop(1006);
    live();
    expect(sockets).toHaveLength(2);
    expect(s.status.chunksReceived).toBe(0);
    s.tick(0);
    expect(log.frames.filter((f) => f !== null)).toEqual([]);
    sock().emit(chunk(10, { seq: 0 })); // seq restarts with the session
    expect(s.status.chunksReceived).toBe(1);
  });
  it("notifies on every status change", () => {
    const seen: string[] = [];
    const k: FakeSocket[] = [];
    const s = createLiveSession({
      url: "x", openSocket: () => { const f = new FakeSocket(); k.push(f); return f; }, every: () => () => {}, defaultCaption: "d",
      sinks: { setFrame() {}, setWorld() {}, setPose() {}, setCaption() {} }, onChange: (st) => seen.push(st.state),
    });
    s.start({ preset: null, rateHz: 0 });
    k[0].open();
    k[0].drop(1006);
    expect(seen).toEqual(["connecting", "starting", "closed"]);
  });
  it("dispose closes quietly", () => {
    const { s, sock, live } = harness();
    live();
    const spy = vi.spyOn(sock(), "close");
    s.dispose();
    expect(spy).toHaveBeenCalledTimes(1);
    expect(s.status.state).toBe("idle");
  });
});

describe("closeMessage and liveCaption", () => {
  it("uses the server's reason for a bad request", () => {
    expect(closeMessage(4400, "seed must be an integer", "live")).toContain("seed must be an integer");
  });
  it("builds the caption from what the server reported and nothing else", () => {
    const t = liveCaption({ speedLabel: "x0.12 real time", backend: "cpu", fallback: null, bodyMode: null, dropped: 0, serverCaption: "cap" });
    expect(t).toBe("live · x0.12 real time · brain: cpu — cap");
    expect(liveCaption({ speedLabel: null, backend: null, fallback: null, bodyMode: null, dropped: 2, serverCaption: "" })).toBe("live · waiting for the first chunk · dropped 2 chunks");
  });
});
