import { describe, expect, it, vi } from "vitest";
import { createWalkPlayer, pickStaticFrame, SLOW_MOTION, WALK_CAPTION, walkCaption } from "../src/bodyWalk";
import { parseWalkClip, type WalkClip } from "../src/bodyClip";

const Q = 10000;
const LEGS = ["lf", "lm", "lh", "rf", "rm", "rh"];

/** 2 Hz, 4 frames (2 s); body "lf_coxa" slides 0, 100, 200, 300 um along fly x so the matrix tells the frame. */
function makeClip(adhesion: number[][] = [[0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0]]): WalkClip {
  const frames = [0, 100, 200, 300].map((x) => [[x, 0, 0, Q, 0, 0, 0]]);
  return parseWalkClip({
    format: "fly-walk/1", quat_scale: Q, thorax_origin_um: [0, 0, 0], bodies: ["lf_coxa"], legs: LEGS, sample_rate_hz: 2,
    n_frames: 4, neutral: { pos_um: [[0, 0, 0]], quat: [[Q, 0, 0, 0]] }, frames, adhesion, contact: adhesion,
  });
}
const zOf = (m: ReadonlyMap<string, ArrayLike<number>> | null) => (m ? m.get("lf_coxa")![14] : null);

function harness(opts: { reducedMotion?: boolean; load?: () => Promise<WalkClip>; clip?: WalkClip; slowdown?: number | null } = {}) {
  const frames: Array<ReadonlyMap<string, ArrayLike<number>> | null> = [];
  const captions: string[] = [];
  const changes: boolean[] = [];
  let pending: { id: number; cb: (ts: number) => void } | null = null;
  let nextId = 1;
  const cancel = vi.fn((id: number) => { if (pending?.id === id) pending = null; });
  const clip = opts.clip ?? makeClip();
  const player = createWalkPlayer({
    load: opts.load ?? (async () => clip),
    setFrame: (m) => frames.push(m),
    setCaption: (t) => captions.push(t),
    defaultCaption: "default",
    schedule: (cb) => { pending = { id: nextId++, cb }; return pending.id; },
    cancel,
    reducedMotion: opts.reducedMotion,
    onChange: (p) => changes.push(p),
    // the timing tests below read real time (1x); null leaves the option out to exercise the default
    ...(opts.slowdown === null ? {} : { slowdown: opts.slowdown ?? 1 }),
  });
  const flush = () => new Promise((r) => setTimeout(r, 0));
  const tick = (ts: number) => { const p = pending; pending = null; p?.cb(ts); };
  return { player, frames, captions, changes, cancel, flush, tick, hasPending: () => pending !== null };
}

describe("WALK_CAPTION", () => {
  it("says exactly what it is, with no forbidden words", () => {
    expect(WALK_CAPTION).toBe("NeuroMechFly v2 physics (flygym 2.1.1): tripod walking controller, not yet driven by the brain");
    expect(WALK_CAPTION).not.toMatch(/\b(die|dies|dead|death|overdose|poison|kill|suffer|pain|feel)\b/i);
  });
});

describe("slow motion", () => {
  it("defaults to real time (the user's choice after trying 20x)", async () => {
    expect(SLOW_MOTION).toBe(1);
    const h = harness({ slowdown: null });
    await h.player.start();
    expect(h.captions).toEqual([WALK_CAPTION]);
    h.tick(0);
    h.tick(250); // 0.25 s = half a frame at 2 Hz -> 50 um
    expect(zOf(h.frames.at(-1)!)).toBeCloseTo(0.05, 9);
  });
  it("a slowdown stretches time and is said in the caption while it plays", async () => {
    const h = harness({ slowdown: 20 });
    await h.player.start();
    expect(h.captions).toEqual([walkCaption(20)]);
    h.tick(0);
    h.tick(5000); // 5 s of wall time at 20x = 0.25 s of clip -> 50 um
    expect(zOf(h.frames.at(-1)!)).toBeCloseTo(0.05, 9);
    expect(walkCaption(20)).toBe(`${WALK_CAPTION}; slow motion x20 (a real fly steps about 12 times a second)`);
    expect(walkCaption(1)).toBe(WALK_CAPTION);
    expect(walkCaption(20)).not.toMatch(/\b(die|dies|dead|death|overdose|poison|kill|suffer|pain|feel)\b/i);
  });
  it("treats a non-finite or non-positive slowdown as real time", async () => {
    for (const bad of [0, -3, Number.NaN, Number.POSITIVE_INFINITY]) {
      const h = harness({ slowdown: bad });
      await h.player.start();
      h.tick(0);
      h.tick(250);
      expect(zOf(h.frames.at(-1)!)).toBeCloseTo(0.05, 9);
      expect(h.captions).toEqual([WALK_CAPTION]);
    }
  });
});

describe("pickStaticFrame", () => {
  it("is the middle of the first run of frames with exactly three legs in stance", () => {
    const clip = makeClip([[0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0]]);
    expect(pickStaticFrame(clip)).toBe(2); // run = frames 1..3, middle = 1 + floor(3 / 2)
  });
  it("falls back to frame 0 without usable flags", () => {
    expect(pickStaticFrame(makeClip([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]))).toBe(0);
    const noFlags = parseWalkClip({ ...JSON.parse(JSON.stringify({
      format: "fly-walk/1", quat_scale: Q, thorax_origin_um: [0, 0, 0], bodies: ["lf_coxa"], sample_rate_hz: 2, n_frames: 1,
      neutral: { pos_um: [[0, 0, 0]], quat: [[Q, 0, 0, 0]] }, frames: [[[0, 0, 0, Q, 0, 0, 0]]],
    })) });
    expect(pickStaticFrame(noFlags)).toBe(0);
  });
});

describe("createWalkPlayer", () => {
  it("starts: loads once, shows the honest caption, and animates from the first tick", async () => {
    const h = harness();
    const started = h.player.start();
    expect(h.player.playing).toBe(true);
    expect(h.changes).toEqual([true]);
    await started;
    expect(h.captions).toEqual([WALK_CAPTION]);
    expect(h.hasPending()).toBe(true);
    h.tick(1000); // t0
    expect(zOf(h.frames.at(-1)!)).toBeCloseTo(0, 9);
    h.tick(1250); // 0.25 s = half a frame at 2 Hz -> 50 um
    expect(zOf(h.frames.at(-1)!)).toBeCloseTo(0.05, 9);
    expect(h.hasPending()).toBe(true);
  });
  it("loops: past the end of the clip the motion wraps to the start", async () => {
    const h = harness();
    await h.player.start();
    h.tick(0);
    h.tick(2000); // exactly one loop (4 frames at 2 Hz)
    expect(zOf(h.frames.at(-1)!)).toBeCloseTo(0, 9);
    h.tick(2500);
    expect(zOf(h.frames.at(-1)!)).toBeCloseTo(0.1, 9);
  });
  it("stops on a second toggle: cancels the frame callback, returns the body to rest and the default caption", async () => {
    const h = harness();
    await h.player.toggle();
    h.tick(0);
    expect(h.player.playing).toBe(true);
    await h.player.toggle();
    expect(h.player.playing).toBe(false);
    expect(h.hasPending()).toBe(false);
    expect(h.frames.at(-1)).toBeNull();
    expect(h.captions.at(-1)).toBe("default");
    expect(h.changes).toEqual([true, false]);
  });
  it("schedules nothing while stopped, and a stop with nothing playing is a no-op", async () => {
    const h = harness();
    h.player.stop();
    expect(h.frames).toEqual([]);
    expect(h.captions).toEqual([]);
    expect(h.changes).toEqual([]);
    expect(h.hasPending()).toBe(false);
  });
  it("a stop while the clip is still loading wins: nothing is shown afterwards", async () => {
    let release!: (c: WalkClip) => void;
    const h = harness({ load: () => new Promise<WalkClip>((r) => { release = r; }) });
    const started = h.player.start();
    h.player.stop();
    release(makeClip());
    await started;
    expect(h.hasPending()).toBe(false);
    expect(h.captions).toEqual(["default"]);
    expect(h.frames.at(-1)).toBeNull();
    expect(h.player.playing).toBe(false);
  });
  it("a start twice is one start; the clip is loaded once across stop and start", async () => {
    const load = vi.fn(async () => makeClip());
    const h = harness({ load });
    await Promise.all([h.player.start(), h.player.start()]);
    h.player.stop();
    await h.player.start();
    expect(load).toHaveBeenCalledTimes(1);
  });
  it("reports a failed load: back to rest, default caption, one warning, playing false", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const h = harness({ load: async () => { throw new Error("404"); } });
    await h.player.start();
    expect(h.player.playing).toBe(false);
    expect(h.changes).toEqual([true, false]);
    expect(h.captions.at(-1)).toBe("default");
    expect(h.hasPending()).toBe(false);
    expect(warn).toHaveBeenCalledTimes(1);
    warn.mockRestore();
  });
  it("can be retried after a failed load", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    let fail = true;
    const h = harness({ load: async () => { if (fail) throw new Error("x"); return makeClip(); } });
    await h.player.start();
    fail = false;
    await h.player.start();
    expect(h.player.playing).toBe(true);
    expect(h.hasPending()).toBe(true);
    vi.restoreAllMocks();
  });
  it("reduced motion: shows one static mid-stance frame, never schedules a frame, and a second toggle returns to rest", async () => {
    const h = harness({ reducedMotion: true });
    await h.player.toggle();
    expect(h.hasPending()).toBe(false);
    expect(h.frames).toHaveLength(1);
    expect(zOf(h.frames[0])).toBeCloseTo(0.2, 9); // frame 2
    expect(h.captions).toEqual([WALK_CAPTION]);
    expect(h.player.playing).toBe(true);
    await h.player.toggle();
    expect(h.frames.at(-1)).toBeNull();
    expect(h.captions.at(-1)).toBe("default");
    expect(h.player.playing).toBe(false);
  });
  it("dispose stops quietly and later calls do nothing", async () => {
    const h = harness();
    await h.player.start();
    h.player.dispose();
    expect(h.hasPending()).toBe(false);
    const n = h.frames.length;
    await h.player.start();
    h.player.stop();
    expect(h.frames.length).toBe(n);
    expect(h.player.playing).toBe(false);
  });
});
