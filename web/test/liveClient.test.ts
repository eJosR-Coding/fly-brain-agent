import { describe, expect, it, vi } from "vitest";
import { createLiveClient, liveUrl, DEV_PRESET, DEV_RATE_HZ } from "../src/liveClient";
import type { SocketLike } from "../src/liveSession";
import { chunk, header } from "./liveFixtures";

class FakeSocket implements SocketLike {
  sent: Array<Record<string, unknown>> = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: ((ev: { code: number; reason: string }) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;
  closed = 0;
  send(d: string) { this.sent.push(JSON.parse(d)); }
  close() { this.closed++; }
  emit(m: unknown) { this.onmessage?.({ data: JSON.stringify(m) }); }
}

function harness() {
  const sockets: FakeSocket[] = [];
  const calls = { frames: [] as unknown[], sandbox: [] as boolean[], captions: [] as string[], prepared: 0 };
  const raf = { queue: new Map<number, (t: number) => void>(), next: 1 };
  const client = createLiveClient({
    location: { protocol: "http:", host: "localhost:5173" },
    openSocket: () => { const s = new FakeSocket(); sockets.push(s); return s; },
    body: {
      setFrame: (m) => calls.frames.push(m), setWorld() {}, setPose() {}, setCaption: (t) => calls.captions.push(t),
      setSandbox: (on) => calls.sandbox.push(on),
    },
    defaultCaption: "default",
    prepare: () => { calls.prepared++; },
    schedule: (cb) => { const id = raf.next++; raf.queue.set(id, cb); return id; },
    cancel: (id) => { raf.queue.delete(id); },
    every: () => () => {},
  });
  const frame = (t: number) => { const cbs = [...raf.queue.values()]; raf.queue.clear(); cbs.forEach((cb) => cb(t)); };
  const live = () => {
    client.start(DEV_PRESET, DEV_RATE_HZ);
    const s = sockets.at(-1)!;
    s.onopen?.();
    s.emit({ type: "started", header: header(), backend: "gpu", caption: "cap" });
    return s;
  };
  return { client, sockets, calls, raf, frame, live };
}

describe("liveUrl", () => {
  it("is the API's live socket on the page's own host, secure when the page is", () => {
    expect(liveUrl({ protocol: "http:", host: "localhost:5173" })).toBe("ws://localhost:5173/api/ws/live");
    expect(liveUrl({ protocol: "https:", host: "fly.example" })).toBe("wss://fly.example/api/ws/live");
  });
});

describe("createLiveClient (dev entry: dng100 at 50 Hz)", () => {
  it("defaults to the dng100 preset at 50 Hz", () => {
    expect([DEV_PRESET, DEV_RATE_HZ]).toEqual(["dng100", 50]);
  });
  it("prepares the stage, opens the sandbox and starts with the preset", () => {
    const { client, calls, live } = harness();
    const s = live();
    expect(calls.prepared).toBe(1);
    expect(calls.sandbox).toEqual([true]);
    expect(s.sent[0]).toEqual({ type: "start", preset: "dng100", rate_hz: 50 });
    expect(client.status.state).toBe("live");
  });
  it("draws from requestAnimationFrame only while live, and stops asking when it ends", () => {
    const { client, raf, frame, sockets, live, calls } = harness();
    expect(raf.queue.size).toBe(0);
    const s = live();
    expect(raf.queue.size).toBe(1);
    for (let i = 0; i < 3; i++) s.emit(chunk(10, { seq: i }));
    frame(0); frame(20);
    expect(calls.frames.length).toBeGreaterThan(0);
    expect(raf.queue.size).toBe(1);
    client.stop();
    expect(raf.queue.size).toBe(0);
    expect(sockets[0].closed).toBe(1);
  });
  it("closes the sandbox again when the session ends, from stop or from a server close", () => {
    const { client, calls, live } = harness();
    live();
    client.stop();
    expect(calls.sandbox).toEqual([true, false]);
    const s = live();
    s.onclose?.({ code: 4409, reason: "busy" });
    expect(calls.sandbox).toEqual([true, false, true, false]);
    expect(calls.captions.at(-1)).toMatch(/another live session/);
  });
  it("toggle starts, then stops", () => {
    const { client, sockets } = harness();
    client.toggle();
    expect(client.status.state).toBe("connecting");
    client.toggle();
    expect(client.status.state).toBe("idle");
    expect(sockets).toHaveLength(1);
  });
  it("is quiet to stop when nothing runs", () => {
    const { client, calls } = harness();
    client.stop();
    expect(calls.sandbox).toEqual([]);
    expect(calls.prepared).toBe(0);
  });
  it("does not prepare the stage again for a refused second start", () => {
    const { client, calls, live } = harness();
    live();
    client.start("dng100", 50);
    expect(calls.prepared).toBe(1);
  });
  it("tells listeners about every state change", () => {
    const seen: string[] = [];
    const { client, live } = harness();
    client.onChange((s) => seen.push(s.state));
    live();
    client.stop();
    expect(seen).toEqual(["connecting", "starting", "live", "idle"]);
  });
  it("dispose cancels the draw loop", () => {
    const { client, raf, live } = harness();
    live();
    const spy = vi.fn();
    client.onChange(spy);
    client.dispose();
    expect(raf.queue.size).toBe(0);
  });
});
