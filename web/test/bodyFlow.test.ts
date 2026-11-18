import { describe, expect, it, vi } from "vitest";
import { BODY_ERRORS, createBodyFlow, driveIsZero, FLOW_PRESETS, NO_MOVEMENT, computingText } from "../src/bodyFlow";

const Q = 10000;
const clipJson = () => ({
  format: "fly-walk/1", quat_scale: Q, thorax_origin_um: [0, 0, 0], bodies: ["lf_coxa"], sample_rate_hz: 2, n_frames: 2, loop: false,
  neutral: { pos_um: [[0, 0, 0]], quat: [[Q, 0, 0, 0]] }, frames: [[[0, 0, 0, Q, 0, 0, 0]], [[0, 0, 0, Q, 0, 0, 0]]],
});
const bodyJson = (drive: unknown = { drive: [[0.5, 0.5]], escape: [false] }) => ({ clip: clipJson(), drive, caption: "OUR mapping caption", note: "n" });
const simJson = (id = "sim1") => ({ simulation_id: id, active_neurons: 5, stimulus: { duration_ms: 1000 } });

type Resp = { ok: boolean; status: number; json: () => Promise<unknown> };
const resp = (status: number, body: unknown): Resp => ({ ok: status >= 200 && status < 300, status, json: async () => body });

/** A fake network: each call takes the next queued answer (a value, a rejection, or a deferred the test resolves). */
function harness(answers: Array<Resp | Error | Promise<Resp>>, opts: { shown?: boolean } = {}) {
  const calls: Array<{ url: string; body: unknown; signal?: AbortSignal }> = [];
  const q = [...answers];
  const status: Array<string | null> = [];
  const states: string[] = [];
  const order: string[] = [];
  const flow = createBodyFlow({
    request: async (url, init) => {
      calls.push({ url, body: init.body ? JSON.parse(init.body) : undefined, signal: init.signal });
      const a = q.shift();
      if (a === undefined) throw new Error("unexpected request");
      if (a instanceof Error) throw a;
      return a;
    },
    showRecording: async (id) => { order.push(`show:${id}`); return opts.shown ?? true; },
    play: (clip, caption) => { order.push(`play:${caption}`); void clip; },
    clearBody: () => order.push("clear"),
    setStatus: (t) => status.push(t),
    onState: (s) => states.push(s),
  });
  return { flow, calls, status, states, order };
}
const flush = () => new Promise((r) => setTimeout(r, 0));

describe("pure helpers", () => {
  it("driveIsZero: all-zero drive and no escape; malformed means unknown (not zero)", () => {
    expect(driveIsZero({ drive: [[0, 0], [0, 0]], escape: [false, false] })).toBe(true);
    expect(driveIsZero({ drive: [[0, 0.2]], escape: [false] })).toBe(false);
    expect(driveIsZero({ drive: [[0, 0]], escape: [true] })).toBe(false);
    expect(driveIsZero({ drive: [[0, 0]] })).toBe(true);
    expect(driveIsZero(null)).toBe(false);
    expect(driveIsZero({ drive: "x" })).toBe(false);
  });
  it("states the cost: ~8 s of compute per simulated second, capped at 3 s", () => {
    expect(computingText(1000)).toBe("computing body physics ~8 s");
    expect(computingText(500)).toBe("computing body physics ~4 s");
    expect(computingText(10000)).toBe("computing body physics ~24 s");
  });
  it("has the three chips with the settings that worked end to end", () => {
    expect(FLOW_PRESETS.walk.run).toEqual({ preset: "dng100", rate_hz: 50, duration_ms: 1000, trials: 3 });
    expect(FLOW_PRESETS.turn_left.run).toEqual({ preset: "dna02_left", rate_hz: 200, duration_ms: 1000, trials: 3 });
    expect(FLOW_PRESETS.turn_right.run).toEqual({ preset: "dna02_right", rate_hz: 200, duration_ms: 1000, trials: 3 });
    expect(FLOW_PRESETS.walk.label).toBe("walk (DNg100)");
    expect(FLOW_PRESETS.turn_left.label).toBe("turn left (DNa02 L)");
    expect(FLOW_PRESETS.turn_right.label).toBe("turn right (DNa02 R)");
  });
});

describe("runPreset", () => {
  it("simulates, computes the body, shows the recording then plays the clip with the route's caption", async () => {
    const h = harness([resp(200, simJson()), resp(200, bodyJson())]);
    await h.flow.runPreset("walk");
    expect(h.calls.map((c) => c.url)).toEqual(["/api/simulate", "/api/body/from-simulation/sim1"]);
    expect(h.calls[0].body).toEqual({ preset: "dng100", rate_hz: 50, duration_ms: 1000, trials: 3 });
    expect(h.status).toContain("computing body physics ~8 s");
    expect(h.order).toEqual(["show:sim1", "play:OUR mapping caption"]);
    expect(h.flow.state).toBe("playing");
    expect(h.states).toEqual(["simulating", "computing", "playing"]);
  });
  it.each([["turn_left", "dna02_left"], ["turn_right", "dna02_right"]] as const)("%s runs the %s preset", async (key, preset) => {
    const h = harness([resp(200, simJson()), resp(200, bodyJson())]);
    await h.flow.runPreset(key);
    expect(h.calls[0].body).toMatchObject({ preset, rate_hz: 200 });
  });
  it("an all-zero drive plays nothing: the recording is shown and the status says why", async () => {
    const h = harness([resp(200, simJson()), resp(200, bodyJson({ drive: [[0, 0], [0, 0]], escape: [false, false] }))]);
    await h.flow.runPreset("walk");
    expect(h.order).toEqual(["show:sim1"]);
    expect(h.status.at(-1)).toBe(NO_MOVEMENT);
    expect(NO_MOVEMENT).toBe("no movement command from the brain (descending neurons below threshold)");
    expect(h.flow.state).toBe("idle");
  });
  it("503 (extra missing) shows the install hint once, no retry, brain recording still shown", async () => {
    const h = harness([resp(200, simJson()), resp(503, { detail: "install it: uv sync --extra body" })]);
    await h.flow.runPreset("walk");
    expect(h.calls.length).toBe(2);
    expect(h.status.at(-1)).toBe(`${BODY_ERRORS.unavailable}: install it: uv sync --extra body`);
    expect(h.order).toEqual(["show:sim1"]);
    expect(h.flow.state).toBe("idle");
  });
  it("404 says the recording expired; 504 says it timed out; 422 relays the detail", async () => {
    for (const [code, text] of [[404, BODY_ERRORS.expired], [504, BODY_ERRORS.timeout], [422, "bad duration"]] as const) {
      const h = harness([resp(200, simJson()), resp(code, { detail: "bad duration" })]);
      await h.flow.runPreset("walk");
      expect(h.status.at(-1)).toBe(text);
      expect(h.flow.state).toBe("idle");
    }
  });
  it("a failing simulation stops before the body is requested", async () => {
    const h = harness([resp(500, { detail: "boom" })]);
    await h.flow.runPreset("walk");
    expect(h.calls.length).toBe(1);
    expect(h.status.at(-1)).toBe("simulation failed: boom");
    expect(h.order).toEqual([]);
  });
  it("a network error is reported, not thrown", async () => {
    const h = harness([resp(200, simJson()), new Error("offline")]);
    await expect(h.flow.runPreset("walk")).resolves.toBeUndefined();
    expect(h.status.at(-1)).toBe("body simulation failed: offline");
  });
  it("a malformed clip is reported", async () => {
    const h = harness([resp(200, simJson()), resp(200, { clip: { format: "nope" }, drive: { drive: [[1, 1]] }, caption: "c" })]);
    await h.flow.runPreset("walk");
    expect(h.status.at(-1)).toMatch(/^body simulation failed: /);
    expect(h.order).not.toContain("play:c");
  });
  it("does not play when the recording was superseded while it loaded", async () => {
    const h = harness([resp(200, simJson()), resp(200, bodyJson())], { shown: false });
    await h.flow.runPreset("walk");
    expect(h.order).toEqual(["show:sim1"]);
    expect(h.flow.state).toBe("idle");
  });
  it("ignores a second press while busy", async () => {
    let release!: (r: Resp) => void;
    const h = harness([new Promise<Resp>((r) => { release = r; }), resp(200, bodyJson())]);
    const first = h.flow.runPreset("walk");
    await h.flow.runPreset("turn_left");
    expect(h.calls.length).toBe(1);
    release(resp(200, simJson()));
    await first;
    expect(h.calls.length).toBe(2);
  });
});

describe("stop / cancel", () => {
  it("stop while simulating aborts the request and drops its answer", async () => {
    let release!: (r: Resp) => void;
    const h = harness([new Promise<Resp>((r) => { release = r; })]);
    const run = h.flow.runPreset("walk");
    await flush();
    h.flow.stop();
    expect(h.calls[0].signal?.aborted).toBe(true);
    expect(h.flow.state).toBe("idle");
    release(resp(200, simJson()));
    await run;
    expect(h.calls.length).toBe(1); // never went on to the body
    expect(h.order).toEqual(["clear"]);
  });
  it("stop while the body computes drops the clip", async () => {
    let release!: (r: Resp) => void;
    const h = harness([resp(200, simJson()), new Promise<Resp>((r) => { release = r; })]);
    const run = h.flow.runPreset("walk");
    await flush();
    expect(h.flow.state).toBe("computing");
    h.flow.stop();
    expect(h.calls[1].signal?.aborted).toBe(true);
    release(resp(200, bodyJson()));
    await run;
    expect(h.order).toEqual(["clear"]);
    expect(h.status.at(-1)).toBeNull();
  });
  it("an abort rejection is silent (no error text)", async () => {
    const abort = Object.assign(new Error("aborted"), { name: "AbortError" });
    const h = harness([abort]);
    await h.flow.runPreset("walk");
    expect(h.status.filter((s) => s && /failed/.test(s))).toEqual([]);
  });
  it("stop while playing clears the body and allows a new run", async () => {
    const h = harness([resp(200, simJson()), resp(200, bodyJson()), resp(200, simJson("sim2")), resp(200, bodyJson())]);
    await h.flow.runPreset("walk");
    h.flow.stop();
    expect(h.flow.state).toBe("idle");
    expect(h.order.at(-1)).toBe("clear");
    await h.flow.runPreset("turn_left");
    expect(h.flow.state).toBe("playing");
  });
  it("a finished clip does not block the next run", async () => {
    const h = harness([resp(200, simJson()), resp(200, bodyJson()), resp(200, simJson("sim2")), resp(200, bodyJson())]);
    await h.flow.runPreset("walk");
    await h.flow.runPreset("turn_right");
    expect(h.calls.length).toBe(4);
    expect(h.order.filter((o) => o.startsWith("play"))).toHaveLength(2);
  });
  it("stop when idle is quiet", () => {
    const h = harness([]);
    h.flow.stop();
    expect(h.order).toEqual([]);
  });
  it("supersede (another recording took the stage) aborts in flight without clearing the body", async () => {
    let release!: (r: Resp) => void;
    const h = harness([resp(200, simJson()), new Promise<Resp>((r) => { release = r; })]);
    const run = h.flow.runPreset("walk");
    await flush();
    h.flow.supersede();
    release(resp(200, bodyJson()));
    await run;
    expect(h.order).toEqual([]);
    expect(h.flow.state).toBe("idle");
  });
  it("supersede is ignored while the flow itself is showing its recording", async () => {
    const calls: string[] = [];
    let flowRef!: ReturnType<typeof createBodyFlow>;
    flowRef = createBodyFlow({
      request: vi.fn().mockResolvedValueOnce(resp(200, simJson())).mockResolvedValueOnce(resp(200, bodyJson())),
      showRecording: async () => { flowRef.supersede(); return true; },
      play: () => calls.push("play"),
      clearBody: () => {}, setStatus: () => {},
    });
    await flowRef.runPreset("walk");
    expect(calls).toEqual(["play"]);
  });
});

describe("moveBody (any other recording)", () => {
  it("skips the simulation, states the cost, and replays the recording together with the clip", async () => {
    const h = harness([resp(200, bodyJson())]);
    await h.flow.moveBody("rec9", 2000);
    expect(h.calls.map((c) => c.url)).toEqual(["/api/body/from-simulation/rec9"]);
    expect(h.status).toContain("computing body physics ~16 s");
    expect(h.order).toEqual(["show:rec9", "play:OUR mapping caption"]);
  });
  it("caps the stated cost at the route's 3 s limit", async () => {
    const h = harness([resp(200, bodyJson())]);
    await h.flow.moveBody("rec9", 9000);
    expect(h.status).toContain("computing body physics ~24 s");
    expect(h.calls[0].body).toEqual({ duration_ms: 3000 });
  });
  it("a zero drive says there is no movement command and plays nothing", async () => {
    const h = harness([resp(200, bodyJson({ drive: [[0, 0]], escape: [false] }))]);
    await h.flow.moveBody("rec9", 1000);
    expect(h.status.at(-1)).toBe(NO_MOVEMENT);
    expect(h.order).toEqual([]);
  });
  it("404 (expired recording)", async () => {
    const h = harness([resp(404, { detail: "unknown simulation" })]);
    await h.flow.moveBody("rec9", 1000);
    expect(h.status.at(-1)).toBe(BODY_ERRORS.expired);
  });
});
