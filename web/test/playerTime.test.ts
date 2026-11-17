// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Activity } from "../src/activity";
import { createPlayer, type Player } from "../src/player";
import type { BrainScene } from "../src/scene";

const activity = (durationMs = 100): Activity => ({
  rows: new Uint32Array([1, 2]),
  counts: new Uint8Array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
  nRows: 2,
  nBins: 10,
  binMs: durationMs / 10,
  durationMs,
  peak: 1,
  activeCount: null,
  id: "sim",
  trials: 1,
  compareId: null,
  silencedRows: null,
  readout: null,
});

const fakeScene = (): BrainScene =>
  ({ setActivity() {}, setActivityFrame() {}, setActivityShapes() {} }) as unknown as BrainScene;

const html = `<div id="tl-modes"></div><button id="tl-play"></button>
  <input id="tl-bar" type="range" min="0" max="100" value="0" /><span id="tl-time"></span><div id="tl-scale"></div>`;

// No real rAF in jsdom tests: frames are queued and stepped by hand with a chosen timestamp.
let queue: Array<(t: number) => void> = [];
const step = (now: number) => {
  const cbs = queue;
  queue = [];
  for (const cb of cbs) cb(now);
};

beforeEach(() => {
  queue = [];
  vi.stubGlobal("requestAnimationFrame", (cb: (t: number) => void) => queue.push(cb));
  vi.stubGlobal("cancelAnimationFrame", () => {
    queue = [];
  });
  vi.spyOn(performance, "now").mockReturnValue(0);
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const make = (opts: { reducedMotion?: boolean; onTime?: (ms: number | null) => void } = {}): { p: Player; root: HTMLElement } => {
  const root = document.createElement("div");
  root.innerHTML = html;
  document.body.appendChild(root);
  const p = createPlayer({ scene: fakeScene(), root, reducedMotion: opts.reducedMotion ?? false, speed: 1, onTime: opts.onTime });
  return { p, root };
};

describe("player.subscribeTime", () => {
  it("receives exactly what onTime receives, and onTime still works", () => {
    const viaOpt: Array<number | null> = [];
    const viaSub: Array<number | null> = [];
    const { p } = make({ onTime: (ms) => viaOpt.push(ms) });
    p.load(activity());
    viaOpt.length = 0;
    p.subscribeTime((ms) => viaSub.push(ms));
    step(20);
    step(60);
    expect(viaOpt.length).toBeGreaterThan(1);
    expect(viaSub).toEqual(viaOpt);
  });

  it("scrubbing reaches subscribers too", () => {
    const got: Array<number | null> = [];
    const { p, root } = make();
    p.load(activity());
    p.subscribeTime((ms) => got.push(ms));
    const bar = root.querySelector<HTMLInputElement>("#tl-bar")!;
    bar.value = "5"; // max is nBins (10): halfway
    bar.dispatchEvent(new Event("input"));
    expect(got).toEqual([50]);
  });

  it("unsubscribe stops delivery and two subscribers are independent", () => {
    const a: Array<number | null> = [];
    const b: Array<number | null> = [];
    const { p } = make();
    p.load(activity());
    const offA = p.subscribeTime((ms) => a.push(ms));
    p.subscribeTime((ms) => b.push(ms));
    step(10);
    offA();
    offA(); // idempotent
    step(20);
    expect(a).toEqual([10]);
    expect(b).toEqual([10, 20]);
  });

  it("a throwing subscriber does not break the others, onTime or the draw loop", () => {
    const got: Array<number | null> = [];
    const viaOpt: Array<number | null> = [];
    const { p } = make({ onTime: (ms) => viaOpt.push(ms) });
    vi.spyOn(console, "error").mockImplementation(() => {});
    p.load(activity());
    viaOpt.length = 0;
    p.subscribeTime(() => {
      throw new Error("boom");
    });
    p.subscribeTime((ms) => got.push(ms));
    step(10);
    expect(got).toEqual([10]);
    expect(viaOpt).toEqual([10]);
    expect(queue.length).toBe(1); // the loop kept going
  });

  it("clear() emits null to every subscriber", () => {
    const a: Array<number | null> = [];
    const b: Array<number | null> = [];
    const { p } = make();
    p.load(activity());
    p.subscribeTime((ms) => a.push(ms));
    p.subscribeTime((ms) => b.push(ms));
    p.clear();
    expect(a).toEqual([null]);
    expect(b).toEqual([null]);
  });

  it("with no recording a new subscriber gets null once", () => {
    const got: Array<number | null> = [];
    const { p } = make();
    p.subscribeTime((ms) => got.push(ms));
    expect(got).toEqual([null]);
    expect(p.currentTime()).toBeNull();
  });

  it("under reduced motion a new subscriber gets the final time once", () => {
    const got: Array<number | null> = [];
    const { p } = make({ reducedMotion: true });
    p.load(activity(250));
    p.subscribeTime((ms) => got.push(ms));
    expect(got).toEqual([250]);
    expect(p.currentTime()).toBe(250);
    expect(queue.length).toBe(0); // no frames are scheduled
  });

  it("a subscriber that unsubscribes inside its callback is safe and does not skip the next one", () => {
    const second: Array<number | null> = [];
    const first: Array<number | null> = [];
    const { p } = make();
    p.load(activity());
    const off = p.subscribeTime((ms) => {
      first.push(ms);
      off();
    });
    p.subscribeTime((ms) => second.push(ms));
    step(10);
    step(20);
    expect(first).toEqual([10]);
    expect(second).toEqual([10, 20]);
  });

  it("currentTime follows the playhead and rests at null after clear", () => {
    const { p } = make();
    p.load(activity());
    expect(p.currentTime()).toBe(0);
    step(30);
    expect(p.currentTime()).toBe(30);
    p.clear();
    expect(p.currentTime()).toBeNull();
  });
});
