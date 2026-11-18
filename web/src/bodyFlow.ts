/** The brain -> body flow of the viewer: run a descending-neuron preset (or take any recording), ask the server for the
 *  physics body it drives, then show recording and body together. A small state machine; the network, the stage and the
 *  caption are injected, so it runs under test with fakes (no DOM, no timers).
 *
 *  Honesty: the clip comes from OUR mapping of descending-neuron rates onto NeuroMechFly's walking controller, and the
 *  caption the route returns says so. No clip is faked: an all-zero drive is reported as "no movement command". */
import { parseSimulation } from "./activity";
import { parseWalkClip, type WalkClip } from "./bodyClip";

export type FlowKey = "walk" | "turn_left" | "turn_right";
export type FlowState = "idle" | "simulating" | "computing" | "playing";

/** The direct-activation presets (like optogenetics) with the rate and length that worked end to end. */
export const FLOW_PRESETS: Record<FlowKey, { label: string; title: string; run: { preset: string; rate_hz: number; duration_ms: number; trials: number } }> = {
  walk: {
    label: "walk (DNg100)",
    title: "activate the forward-walking descending neurons DNg100 (50 Hz, 1 s), then compute the physics body they would drive; the mapping from neurons to walking is ours",
    run: { preset: "dng100", rate_hz: 50, duration_ms: 1000, trials: 3 },
  },
  turn_left: {
    label: "turn left (DNa02 L)",
    title: "activate the left DNa02 steering neuron (200 Hz, 1 s), then compute the physics body it would drive; the mapping from neurons to steering is ours",
    run: { preset: "dna02_left", rate_hz: 200, duration_ms: 1000, trials: 3 },
  },
  turn_right: {
    label: "turn right (DNa02 R)",
    title: "activate the right DNa02 steering neuron (200 Hz, 1 s), then compute the physics body it would drive; the mapping from neurons to steering is ours",
    run: { preset: "dna02_right", rate_hz: 200, duration_ms: 1000, trials: 3 },
  },
};

/** The route's length cap and measured cost (see flybrain.body_service). */
export const BODY_MAX_MS = 3000;
const WALL_S_PER_SIM_S = 8;

export const NO_MOVEMENT = "no movement command from the brain (descending neurons below threshold)";
export const BODY_ERRORS = {
  unavailable: "body physics is not installed on this server",
  expired: "that recording is no longer available on the server; run it again",
  timeout: "the body simulation timed out; try again",
} as const;

export const computingText = (durationMs: number): string =>
  `computing body physics ~${Math.ceil((Math.min(durationMs, BODY_MAX_MS) / 1000) * WALL_S_PER_SIM_S)} s`;

/** True when the drive says nothing moves: every left/right value is 0 and no escape. A drive that cannot be read is
 *  NOT called zero (the clip is then shown as it is). */
export function driveIsZero(drive: unknown): boolean {
  if (typeof drive !== "object" || drive === null) return false;
  const d = drive as { drive?: unknown; escape?: unknown };
  if (!Array.isArray(d.drive) || !d.drive.every((row) => Array.isArray(row) && row.every((v) => typeof v === "number" && Number.isFinite(v)))) return false;
  if (d.escape !== undefined && !Array.isArray(d.escape)) return false;
  const anyEscape = (d.escape as unknown[] | undefined)?.some(Boolean) ?? false;
  return !anyEscape && (d.drive as number[][]).every((row) => row.every((v) => Math.abs(v) < 1e-9));
}

export interface FlowResponse { ok: boolean; status: number; json(): Promise<unknown> }
export interface FlowDeps {
  request(url: string, init: { method: string; headers?: Record<string, string>; body?: string; signal?: AbortSignal }): Promise<FlowResponse>;
  /** Load and show the recording (brain wave); false when something else took the stage meanwhile. */
  showRecording(simulationId: string): Promise<boolean>;
  /** Hand the clip to the body (after the recording is shown), with the caption the route returned. */
  play(clip: WalkClip, caption: string): void;
  /** Return the body to rest. */
  clearBody(): void;
  /** Progress or outcome text for the viewer; null clears it. */
  setStatus(text: string | null): void;
  onState?(state: FlowState): void;
}

export interface BodyFlow {
  readonly state: FlowState;
  /** Run a direct-activation preset, compute its body, play both. Ignored while another run is in flight. */
  runPreset(key: FlowKey): Promise<void>;
  /** Compute and play the body of a recording already on the server (any run). Ignored while busy. */
  moveBody(simulationId: string, durationMs: number): Promise<void>;
  /** The user stops: abort what is in flight and return the body to rest. Quiet when idle. */
  stop(): void;
  /** Something else took the stage: abort what is in flight, leave the body alone. Ignored while this flow shows its own recording. */
  supersede(): void;
}

const detailOf = async (r: FlowResponse): Promise<string | null> => {
  const b = (await r.json().catch(() => null)) as { detail?: unknown } | null;
  return typeof b?.detail === "string" ? b.detail : null;
};
const isAbort = (e: unknown) => e instanceof Error && e.name === "AbortError";

export function createBodyFlow(deps: FlowDeps): BodyFlow {
  let state: FlowState = "idle";
  let generation = 0;
  let abort: AbortController | null = null;
  let showing = false;

  const setState = (s: FlowState) => {
    if (state === s) return;
    state = s;
    deps.onState?.(s);
  };
  const post = (url: string, body: unknown, signal: AbortSignal) =>
    deps.request(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal });
  /** a run is in flight (a finished clip on stage is not busy: the next run replaces it) */
  const busy = () => state === "simulating" || state === "computing";
  const halt = () => { generation++; abort?.abort(); abort = null; };

  /** body step shared by both entries; `mine` is this run's generation. `show` false: the recording is already on stage. */
  async function bodyStep(id: string, durationMs: number, mine: number, signal: AbortSignal, show: boolean): Promise<void> {
    const ms = Math.min(durationMs, BODY_MAX_MS);
    setState("computing");
    deps.setStatus(computingText(ms));
    let res: FlowResponse;
    try {
      res = await post(`/api/body/from-simulation/${encodeURIComponent(id)}`, { duration_ms: ms }, signal);
    } catch (e) {
      if (mine !== generation || isAbort(e)) return;
      await fail(`body simulation failed: ${(e as Error).message}`, id, show, mine);
      return;
    }
    if (mine !== generation) return;
    if (!res.ok) {
      const detail = await detailOf(res);
      if (mine !== generation) return;
      if (res.status === 503) return fail(`${BODY_ERRORS.unavailable}${detail ? `: ${detail}` : ""}`, id, show, mine);
      if (res.status === 404) return fail(BODY_ERRORS.expired, id, false, mine);
      if (res.status === 504) return fail(BODY_ERRORS.timeout, id, show, mine);
      return fail(detail ?? `body simulation failed: HTTP ${res.status}`, id, show, mine);
    }
    let clip: WalkClip, caption: string, drive: unknown;
    try {
      const body = (await res.json()) as { clip?: unknown; drive?: unknown; caption?: unknown };
      if (mine !== generation) return;
      clip = parseWalkClip(body.clip);
      drive = body.drive;
      caption = typeof body.caption === "string" ? body.caption : "";
    } catch (e) {
      if (mine !== generation) return;
      return fail(`body simulation failed: ${(e as Error).message}`, id, show, mine);
    }
    if (driveIsZero(drive)) {
      if (show) await showOwn(id);
      if (mine !== generation) return;
      deps.setStatus(NO_MOVEMENT);
      setState("idle");
      return;
    }
    // recording and body start together: the recording (re)starts now and the clip follows its clock
    if (!(await showOwn(id))) { if (mine === generation) { deps.setStatus(null); setState("idle"); } return; }
    if (mine !== generation) return;
    deps.play(clip, caption);
    setState("playing");
  }

  const showOwn = async (id: string): Promise<boolean> => {
    showing = true;
    try { return await deps.showRecording(id); } finally { showing = false; }
  };

  /** Report a failure; the recording still shows when it exists (the brain half worked). */
  async function fail(text: string, id: string, show: boolean, mine: number): Promise<void> {
    if (show) await showOwn(id);
    if (mine !== generation) return;
    deps.setStatus(text);
    setState("idle");
  }

  return {
    get state() { return state; },
    async runPreset(key) {
      if (busy()) return;
      const mine = ++generation;
      const ctl = (abort = new AbortController());
      setState("simulating");
      deps.setStatus(`simulating the brain: ${FLOW_PRESETS[key].label}`);
      let id: string;
      let durationMs: number;
      try {
        const res = await post("/api/simulate", FLOW_PRESETS[key].run, ctl.signal);
        if (!res.ok) {
          const detail = await detailOf(res);
          if (mine !== generation) return;
          deps.setStatus(`simulation failed: ${detail ?? `HTTP ${res.status}`}`);
          setState("idle");
          return;
        }
        const sim = parseSimulation(await res.json());
        if (mine !== generation) return;
        if (typeof sim === "string") throw new Error(`unexpected simulation answer: ${sim}`);
        id = sim.simulationId;
        durationMs = sim.durationMs;
      } catch (e) {
        if (mine !== generation || isAbort(e)) return;
        deps.setStatus(`simulation failed: ${(e as Error).message}`);
        setState("idle");
        return;
      }
      await bodyStep(id, durationMs, mine, ctl.signal, true);
    },
    async moveBody(simulationId, durationMs) {
      if (busy()) return;
      const mine = ++generation;
      const ctl = (abort = new AbortController());
      await bodyStep(simulationId, durationMs, mine, ctl.signal, false);
    },
    stop() {
      if (state === "idle") return;
      halt();
      deps.setStatus(null); // first: clearing the body may then put the recording's own caption back
      deps.clearBody();
      setState("idle");
    },
    supersede() {
      if (showing || state === "idle") return;
      halt();
      setState("idle");
    },
  };
}
