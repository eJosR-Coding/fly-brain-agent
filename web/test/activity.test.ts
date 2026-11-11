import { describe, expect, it } from "vitest";
import {
  activityLegend,
  playLabel,
  binAt,
  createPlayback,
  decodeFrames,
  fillBrightness,
  formatSimulating,
  formatTime,
  loadActivity,
  parseActivityEvent,
  parseSimulation,
  scatterBrightness,
  scrub,
  summedBrightness,
  tick,
  togglePlay,
  type Activity,
} from "../src/activity";

/** Wire format: uint32 LE row indices, then uint8 counts row-major. */
function wire(rows: number[], counts: number[][]): ArrayBuffer {
  const nBins = counts[0]?.length ?? 0;
  const buf = new ArrayBuffer(rows.length * 4 + rows.length * nBins);
  const view = new DataView(buf);
  rows.forEach((r, i) => view.setUint32(i * 4, r, true));
  new Uint8Array(buf, rows.length * 4).set(counts.flat());
  return buf;
}

const act = (counts: number[][], rows = counts.map((_, i) => 10 + i), binMs = 10): Activity =>
  decodeFrames(wire(rows, counts), rows.length, counts[0].length, binMs);

describe("decodeFrames", () => {
  it("reads little-endian rows then row-major counts", () => {
    const a = decodeFrames(wire([5, 300], [[0, 2, 1], [4, 0, 0]]), 2, 3, 10);
    expect(Array.from(a.rows)).toEqual([5, 300]);
    expect(Array.from(a.counts)).toEqual([0, 2, 1, 4, 0, 0]);
    expect(a.nRows).toBe(2);
    expect(a.nBins).toBe(3);
    expect(a.durationMs).toBe(30);
    expect(a.peak).toBe(4);
  });

  it("an empty recording is valid", () => {
    const a = decodeFrames(new ArrayBuffer(0), 0, 100, 10);
    expect(a.nRows).toBe(0);
    expect(a.peak).toBe(1);
  });

  it("rejects a buffer whose size does not match the meta", () => {
    expect(() => decodeFrames(new ArrayBuffer(10), 2, 3, 10)).toThrow(/expected 14 bytes/);
    expect(() => decodeFrames(wire([1], [[1, 2]]), 1, 3, 10)).toThrow(/expected 7 bytes, got 6/);
  });

  it("rejects nonsense dimensions", () => {
    expect(() => decodeFrames(new ArrayBuffer(0), -1, 3, 10)).toThrow();
    expect(() => decodeFrames(new ArrayBuffer(0), 1.5, 3, 10)).toThrow();
    expect(() => decodeFrames(new ArrayBuffer(0), 0, 0, 10)).toThrow();
    expect(() => decodeFrames(new ArrayBuffer(0), 0, 3, 0)).toThrow();
  });
});

describe("fillBrightness", () => {
  it("is 0 where nothing spiked and rises with the count (0..1, peak = 1)", () => {
    const a = act([[0, 1, 4], [0, 0, 0]]);
    const out = new Float32Array(2);
    fillBrightness(a, 2, out, 0);
    expect(out[0]).toBe(1);
    expect(out[1]).toBe(0);
    fillBrightness(a, 1, out, 0);
    expect(out[0]).toBeGreaterThan(0);
    expect(out[0]).toBeLessThan(1);
    fillBrightness(a, 0, out, 0);
    expect(out[0]).toBe(0);
  });

  it("a trail makes the wave linger and decay after a spike", () => {
    const a = act([[4, 0, 0, 0]]);
    const out = new Float32Array(1);
    const seen: number[] = [];
    for (let b = 0; b < 4; b++) {
      fillBrightness(a, b, out, 3);
      seen.push(out[0]);
    }
    expect(seen[0]).toBe(1);
    expect(seen[1]).toBeGreaterThan(seen[2]);
    expect(seen[2]).toBeGreaterThan(seen[3]);
    expect(seen[3]).toBeGreaterThan(0);
  });

  it("clamps the bin into range and never exceeds 1", () => {
    const a = act([[255, 255]]);
    const out = new Float32Array(1);
    fillBrightness(a, 99, out, 2);
    expect(out[0]).toBeLessThanOrEqual(1);
    fillBrightness(a, -5, out, 2);
    expect(out[0]).toBe(1);
  });

  it("refuses an output buffer that is too small", () => {
    expect(() => fillBrightness(act([[1], [1]]), 0, new Float32Array(1))).toThrow();
  });
});

describe("summedBrightness", () => {
  it("is the whole recording in one frame, busiest row = 1", () => {
    const a = act([[1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]);
    const out = new Float32Array(3);
    summedBrightness(a, out);
    expect(out[0]).toBe(1);
    expect(out[1]).toBeGreaterThan(0);
    expect(out[1]).toBeLessThan(1);
    expect(out[2]).toBe(0);
  });
});

describe("playback", () => {
  it("starts playing at 0 and advances by dt * speed", () => {
    let p = createPlayback(1000, { speed: 0.5 });
    expect(p).toMatchObject({ timeMs: 0, playing: true, loop: false, durationMs: 1000 });
    p = tick(p, 200);
    expect(p.timeMs).toBe(100);
  });

  it("plays once by default: stops on the final frame and stays there", () => {
    let p = tick({ ...createPlayback(1000, { speed: 1 }), timeMs: 950 }, 100);
    expect(p).toMatchObject({ timeMs: 1000, playing: false });
    p = tick(p, 5000);
    expect(p).toMatchObject({ timeMs: 1000, playing: false });
  });

  it("loops at the end only when asked", () => {
    const p = tick({ ...createPlayback(1000, { speed: 1, loop: true }), timeMs: 950 }, 100);
    expect(p.timeMs).toBeCloseTo(50);
    expect(p.playing).toBe(true);
  });

  it("without loop it stops at the end", () => {
    const p = tick({ ...createPlayback(1000, { speed: 1, loop: false }), timeMs: 950 }, 100);
    expect(p).toMatchObject({ timeMs: 1000, playing: false });
  });

  it("does not move while paused, and togglePlay resumes (restarting a finished run)", () => {
    let p = togglePlay(createPlayback(1000));
    expect(p.playing).toBe(false);
    expect(tick(p, 500).timeMs).toBe(0);
    p = togglePlay(p);
    expect(p.playing).toBe(true);
    const done = togglePlay({ ...createPlayback(1000, { loop: false }), timeMs: 1000, playing: false });
    expect(done).toMatchObject({ timeMs: 0, playing: true });
  });

  it("replay after the end starts from t = 0 and plays again to the end", () => {
    const ended = tick({ ...createPlayback(1000, { speed: 1 }), timeMs: 999 }, 10);
    expect(playLabel(ended)).toBe("replay");
    const again = togglePlay(ended);
    expect(again).toMatchObject({ timeMs: 0, playing: true });
    expect(tick(again, 400).timeMs).toBe(400);
  });

  it("the scrubber stays usable after the end: scrubbing back pauses on that frame", () => {
    const ended = tick({ ...createPlayback(1000, { speed: 1 }), timeMs: 999 }, 10);
    const back = scrub(ended, 0.25);
    expect(back).toMatchObject({ timeMs: 250, playing: false });
    expect(playLabel(back)).toBe("play");
    expect(togglePlay(back)).toMatchObject({ timeMs: 250, playing: true });
  });

  it("labels the button pause / play / replay", () => {
    expect(playLabel(createPlayback(1000))).toBe("pause");
    expect(playLabel({ ...createPlayback(1000), playing: false, timeMs: 300 })).toBe("play");
    expect(playLabel({ ...createPlayback(1000), playing: false, timeMs: 1000 })).toBe("replay");
  });

  it("scrub sets time from a 0..1 fraction (clamped) and keeps the play state", () => {
    const p = scrub(createPlayback(1000), 0.34);
    expect(p.timeMs).toBeCloseTo(340);
    expect(scrub(p, 2).timeMs).toBe(1000);
    expect(scrub(p, -1).timeMs).toBe(0);
    expect(scrub(p, Number.NaN).timeMs).toBe(0);
    expect(scrub(togglePlay(p), 0.5).playing).toBe(false);
  });

  it("ignores bad dt values", () => {
    const p = createPlayback(1000);
    expect(tick(p, -5).timeMs).toBe(0);
    expect(tick(p, Number.NaN).timeMs).toBe(0);
  });

  it("maps time to a bin index", () => {
    expect(binAt(340, 10, 100)).toBe(34);
    expect(binAt(1000, 10, 100)).toBe(99);
    expect(binAt(-3, 10, 100)).toBe(0);
  });
});

describe("text", () => {
  it("formats the timeline label", () => {
    expect(formatTime(340)).toBe("t = 340 ms");
    expect(formatTime(339.6)).toBe("t = 340 ms");
    expect(formatTime(-1)).toBe("t = 0 ms");
  });

  it("formats the pending text", () => {
    expect(formatSimulating(1000, 0)).toBe("simulating 1 s of brain activity… · 0 s");
    expect(formatSimulating(500, 7.4)).toBe("simulating 0.5 s of brain activity… · 7 s");
  });

  it("formats the legend line, with the readout when there is one", () => {
    expect(activityLegend(336, 375, { label: "MN9", hz: 60.3 })).toBe("activity · 336 active > 1 Hz · 375 spiked · MN9 60 Hz");
    expect(activityLegend(1, 1, null)).toBe("activity · 1 active > 1 Hz · 1 spiked");
    expect(activityLegend(2926, 3000, null)).toBe("activity · 2,926 active > 1 Hz · 3,000 spiked");
    expect(activityLegend(0, 12, null)).toBe("activity · 0 active > 1 Hz · 12 spiked");
  });

  it("without a backend count it only claims what it knows", () => {
    expect(activityLegend(null, 375, null)).toBe("activity · 375 spiked");
  });
});

describe("parseSimulation", () => {
  const ok = { simulation_id: "sim_ab12cd", active_neurons: 336, readout: { mean_hz: 60.3 }, stimulus: { duration_ms: 1000 } };
  it("reads the summary of POST /simulate", () => {
    expect(parseSimulation(ok)).toEqual({ simulationId: "sim_ab12cd", active: 336, readoutHz: 60.3, durationMs: 1000 });
  });
  it("the readout is optional", () => {
    const { readout, ...rest } = ok;
    expect(parseSimulation(rest)).toMatchObject({ readoutHz: null });
  });
  it("rejects a payload without an id", () => {
    expect(parseSimulation({ ...ok, simulation_id: 3 })).toMatch(/simulation_id/);
    expect(parseSimulation(null)).toMatch(/object/);
  });
});

describe("parseActivityEvent", () => {
  it("validates the SSE payload", () => {
    expect(parseActivityEvent({ simulation_id: "sim_1", n_bins: 100, bin_ms: 10 })).toEqual({ simulationId: "sim_1", nBins: 100, binMs: 10, activeCount: null });
    expect(parseActivityEvent({ simulation_id: "sim_1", n_bins: 100, bin_ms: 10, active_count: 336, active_hz: 1 })).toMatchObject({ activeCount: 336 });
    expect(parseActivityEvent({ simulation_id: "sim_1", n_bins: 100, bin_ms: 10, active_count: "x" })).toMatchObject({ activeCount: null });
    expect(parseActivityEvent({ simulation_id: "sim_1", n_bins: 0, bin_ms: 10 })).toMatch(/n_bins/);
    expect(parseActivityEvent({ n_bins: 3, bin_ms: 10 })).toMatch(/simulation_id/);
  });
});

describe("loadActivity", () => {
  it("fetches meta then frames and decodes them", async () => {
    const calls: string[] = [];
    const f = (async (url: string) => {
      calls.push(url);
      if (url.endsWith("/frames")) return new Response(wire([7], [[1, 3]]));
      return Response.json({ simulation_id: "sim_x", bin_ms: 5, n_bins: 2, n_rows: 1, duration_ms: 10, stimulus: {}, trials: 1 });
    }) as unknown as typeof fetch;
    const a = await loadActivity("sim_x", f);
    expect(calls).toEqual(["/api/simulations/sim_x", "/api/simulations/sim_x/frames"]);
    expect(Array.from(a.rows)).toEqual([7]);
    expect(a.binMs).toBe(5);
    expect(a.peak).toBe(3);
    expect(a.activeCount).toBeNull();
  });

  it("carries the backend's active count (> 1 Hz) from the meta", async () => {
    const f = (async (url: string) =>
      url.endsWith("/frames")
        ? new Response(wire([7], [[1, 3]]))
        : Response.json({ bin_ms: 5, n_bins: 2, n_rows: 1, duration_ms: 10, active_count: 336, active_hz: 1 })) as unknown as typeof fetch;
    expect((await loadActivity("sim_x", f)).activeCount).toBe(336);
  });

  it("surfaces the backend message on 404", async () => {
    const f = (async () => Response.json({ detail: "simulation 'sim_x' expired" }, { status: 404 })) as unknown as typeof fetch;
    await expect(loadActivity("sim_x", f)).rejects.toThrow(/expired/);
  });

  it("rejects frames of the wrong size", async () => {
    const f = (async (url: string) =>
      url.endsWith("/frames")
        ? new Response(new ArrayBuffer(3))
        : Response.json({ bin_ms: 5, n_bins: 2, n_rows: 1, duration_ms: 10 })) as unknown as typeof fetch;
    await expect(loadActivity("sim_x", f)).rejects.toThrow(/expected 6 bytes/);
  });
});

describe("scatterBrightness", () => {
  it("writes each recorded row's brightness at its point index and ignores rows outside the cloud", () => {
    const target = new Float32Array(6).fill(9);
    scatterBrightness(target, Uint32Array.from([4, 1, 99]), Float32Array.from([0.5, 1, 0.7]));
    expect(Array.from(target)).toEqual([9, 1, 9, 9, 0.5, 9]);
  });
  it("uses only as many values as there are rows", () => {
    const target = new Float32Array(3);
    scatterBrightness(target, Uint32Array.from([2]), Float32Array.from([0.25, 0.9, 0.9]));
    expect(Array.from(target)).toEqual([0, 0, 0.25]);
  });
});
