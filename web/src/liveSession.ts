/** The client of /ws/live: one live brain+body session, as a small state machine. The socket, the timers and the
 *  sinks (body, caption) are injected, so it runs under test with fakes (no DOM, no clock, no network).
 *
 *  Honesty: the speed label, the brain backend, any GPU->CPU fallback, the body mode and the server's caption are shown
 *  exactly as the server reports them. Playback runs on the chunks' own time base (livePlayback); the viewer never
 *  claims more speed than the server measured. No automatic reconnect: after a close the user starts again.
 *
 *  State is plain fields, not reactive: `tick` runs from requestAnimationFrame (see liveClient) and writes straight to
 *  the sinks. */
import { clipMatrices, escapeAt, wingCycleAt, wingsAt, worldPose } from "./bodyClip";
import type { JointId } from "./bodyPose";
import { escapePose, flightWingPose } from "./bodyWorld";
import { objectToViewer, parseLiveChunk, parseLiveHeader, type LiveChunk, type LiveHeader } from "./liveFrames";
import { createPlayback } from "./livePlayback";
import type { WorldPose } from "./bodyClip";
import { parseArena, type Arena } from "./bodyCube";

/** The slice of the browser's WebSocket the session needs. */
export interface SocketLike {
  send(data: string): void;
  close(code?: number, reason?: string): void;
  onopen: (() => void) | null;
  onmessage: ((ev: { data: unknown }) => void) | null;
  onclose: ((ev: { code: number; reason: string }) => void) | null;
  onerror: ((ev: unknown) => void) | null;
}

export interface LiveSinks {
  setFrame(mats: ReadonlyMap<string, ArrayLike<number>> | null): void;
  setWorld(w: WorldPose | null): void;
  setPose(angles: Partial<Record<JointId, number>>): void;
  setCaption(text: string): void;
  /** The arena the server enforces (null: none sent, draw the fallback cube). Called once per `started`. */
  setArena?(arena: Arena | null): void;
  /** The server's world objects (sugar/coffee cubes), viewer frame: `[{ id, kind, pos_mm: [x, y above floor, z], size_mm }]`. */
  setObjects?(objects: unknown): void;
}

export type LiveState = "idle" | "connecting" | "starting" | "live" | "closed";

export interface LiveStatus {
  state: LiveState;
  /** why the session is not running, or the last server complaint; null when there is nothing to say */
  message: string | null;
  speedLabel: string | null;
  backend: string | null;
  fallback: string | null;
  bodyMode: string | null;
  serverCaption: string;
  note: string;
  chunksReceived: number;
  chunksPlayed: number;
  /** duplicate or out-of-order chunks */
  ignored: number;
  malformed: number;
  dropped: number;
  bufferedMs: number;
}

export interface LiveStartOptions { preset: string | null; rateHz: number; seed?: number; chunkMs?: number; vision?: "arena" }

export interface LiveDeps {
  url: string;
  openSocket(url: string): SocketLike;
  sinks: LiveSinks;
  /** Repeat `fn` every `ms` until the returned function is called (the keepalive). */
  every(fn: () => void, ms: number): () => void;
  /** The caption shown when no live session is running. */
  defaultCaption: string;
  onChange?(status: LiveStatus): void;
  /** keepalive period; the server stops an idle session after 5 minutes */
  pingMs?: number;
  /** longest wall gap one draw may advance playback by (a sleeping tab must not jump ahead) */
  maxDtMs?: number;
}

export interface LiveSession {
  readonly status: LiveStatus;
  /** Open the socket and start. False when a session is already in flight or the socket cannot be created. */
  start(opts: LiveStartOptions): boolean;
  /** Change the drive of the running session. False unless live. */
  stimulus(preset: string | null, rateHz: number): boolean;
  /** End the session (quiet when idle); the body returns to rest. */
  stop(): void;
  /** One draw at wall time `nowMs`. True while the session wants more draws. */
  tick(nowMs: number): boolean;
  dispose(): void;
}

export const PING_MS = 60_000;
const MAX_DT_MS = 100;

/** What the server's close code means to the person looking at the viewer. */
export function closeMessage(code: number, reason: string, state: LiveState | "live"): string {
  const why = reason ? ` (${reason})` : "";
  switch (code) {
    case 1000: return "live session ended";
    case 4400: return `the server rejected the live request: ${reason || "bad request"}`;
    case 4403: return "the server does not allow this page's origin for live sessions";
    case 4408: return "live session stopped: no activity for too long (idle timeout); start a new one";
    case 4409: return "another live session is already running on this server; stop it or wait, then start again";
    case 4503: return `live mode is not available on this server${why}: install the body physics with 'uv sync --extra body' and build the brain weights with 'flybrain-build-sim-weights'`;
    case 1011: return `the live session failed on the server${why}`;
    default:
      return state === "connecting" || state === "starting"
        ? `could not connect to the live server${why}; is the API running?`
        : `the connection to the live server was lost${why}; press live to start again`;
  }
}

/** The line under the body while a session runs: only what the server reported, plus the server's own caption. */
/** display flapping rate of a flying fly (Hz of wall time) */
export const WING_FLAP_HZ = 20;

export function liveCaption(s: Pick<LiveStatus, "speedLabel" | "backend" | "fallback" | "bodyMode" | "dropped" | "serverCaption"> & { surface?: string | null; flightNote?: string }): string {
  const parts = ["live", s.speedLabel ?? "waiting for the first chunk"];
  if (s.backend) parts.push(`brain: ${s.backend}`);
  if (s.fallback) parts.push(`fallback: ${s.fallback}`);
  if (s.bodyMode) parts.push(`body: ${s.bodyMode}`);
  if (s.surface && s.surface !== "floor" && s.surface !== "flight") parts.push(`walking on the ${s.surface === "ceiling" ? "ceiling" : `wall ${s.surface}`} (world rule: adhesive pads)`);
  if (s.dropped > 0) parts.push(`dropped ${s.dropped} chunk${s.dropped === 1 ? "" : "s"}`);
  return parts.join(" · ") + (s.serverCaption ? ` — ${s.serverCaption}` : "") + (s.flightNote ? ` — ${s.flightNote}` : "");
}

export function createLiveSession(deps: LiveDeps): LiveSession {
  const playback = createPlayback<LiveChunk>({ targetMs: 100, maxChunks: 8 });
  let state: LiveState = "idle";
  let message: string | null = null;
  let socket: SocketLike | null = null;
  let generation = 0; // a socket that was stopped or replaced is deaf
  let header: LiveHeader | null = null;
  let z0: number | null = null;
  let lastSeq = -1;
  let stopped = false; // the server said "stopped"
  let cancelPing: (() => void) | null = null;
  let lastTick: number | null = null;
  let shownSeq = -1;
  let shownT = -1;
  let sentEscape: boolean | null = null;
  let sentWings = false;
  let lastCaption: string | null = null;
  let lastObjects: string | null = null;
  let info = { speedLabel: null as string | null, backend: null as string | null, fallback: null as string | null, bodyMode: null as string | null, surface: null as string | null };
  let serverCaption = "";
  let flightNote = ""; // the wing-slowdown label, while the chunk in view carries flight wings
  let note = "";
  const n = { received: 0, played: 0, ignored: 0, malformed: 0 };

  const status = (): LiveStatus => ({
    state, message, ...info, serverCaption, note,
    chunksReceived: n.received, chunksPlayed: n.played, ignored: n.ignored, malformed: n.malformed,
    dropped: playback.dropped, bufferedMs: playback.bufferedMs,
  });
  const notify = () => deps.onChange?.(status());
  const setState = (s: LiveState) => { state = s; notify(); };
  const guard = (what: string, fn: () => void) => {
    try { fn(); } catch (e) { console.error(`live ${what} failed`, e); }
  };
  const caption = (text: string) => {
    if (text === lastCaption) return;
    lastCaption = text;
    guard("caption", () => deps.sinks.setCaption(text));
  };
  const rest = () => {
    guard("frame", () => deps.sinks.setFrame(null));
    guard("world", () => deps.sinks.setWorld(null));
    guard("pose", () => deps.sinks.setPose({}));
    if (lastObjects !== null) guard("objects", () => deps.sinks.setObjects?.([]));
    lastObjects = null;
    sentEscape = false;
    sentWings = false;
  };
  const release = () => { // drop the socket and everything tied to it
    generation++;
    cancelPing?.();
    cancelPing = null;
    const s = socket;
    socket = null;
    if (s) { s.onopen = s.onmessage = s.onclose = s.onerror = null; }
    return s;
  };
  const reset = () => {
    playback.reset();
    header = null; z0 = null; lastSeq = -1; stopped = false; lastTick = null; shownSeq = -1; shownT = -1; sentEscape = null; sentWings = false;
    lastCaption = null; message = null; serverCaption = ""; note = "";
    info = { speedLabel: null, backend: null, fallback: null, bodyMode: null, surface: null };
    n.received = n.played = n.ignored = n.malformed = 0;
  };
  const send = (cmd: Record<string, unknown>): boolean => {
    if (!socket) return false;
    try { socket.send(JSON.stringify(cmd)); return true; } catch (e) { console.error("live send failed", e); return false; }
  };
  const finish = (next: LiveState, text: string | null) => {
    release();
    playback.reset();
    rest();
    message = text;
    caption(text ?? deps.defaultCaption);
    setState(next);
  };
  const refreshCaption = () => caption(liveCaption({ ...info, dropped: playback.dropped, serverCaption, flightNote }));
  const malformed = () => { n.malformed++; };

  const onMessage = (data: unknown) => {
    if (typeof data !== "string") return malformed();
    let msg: Record<string, unknown>;
    try {
      const parsed: unknown = JSON.parse(data);
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return malformed();
      msg = parsed as Record<string, unknown>;
    } catch { return malformed(); }
    switch (msg.type) {
      case "started": {
        if (state !== "starting") return malformed();
        try { header = parseLiveHeader(msg.header); } catch (e) {
          socket?.close(1000, "bad header");
          finish("closed", `the server sent a body header this viewer cannot read: ${(e as Error).message}`);
          return;
        }
        info.backend = typeof msg.backend === "string" ? msg.backend : null;
        info.fallback = typeof msg.fallback === "string" && msg.fallback ? msg.fallback : null;
        serverCaption = typeof msg.caption === "string" ? msg.caption : "";
        note = typeof msg.note === "string" ? msg.note : "";
        guard("arena", () => deps.sinks.setArena?.(parseArena(msg.arena)));
        setState("live");
        refreshCaption();
        cancelPing = deps.every(() => { if (state === "live") send({ type: "ping" }); }, deps.pingMs ?? PING_MS);
        return;
      }
      case "chunk": {
        if (state !== "live" || !header) return malformed();
        let parsed: ReturnType<typeof parseLiveChunk>;
        try { parsed = parseLiveChunk(msg, header, z0); } catch { return malformed(); }
        if (parsed.chunk.seq <= lastSeq) { n.ignored++; return; }
        lastSeq = parsed.chunk.seq;
        z0 = parsed.z0;
        n.received++;
        playback.push(parsed.chunk);
        info = { speedLabel: parsed.chunk.info.speedLabel, backend: parsed.chunk.info.backend ?? info.backend, fallback: parsed.chunk.info.fallback, bodyMode: parsed.chunk.info.bodyMode, surface: parsed.chunk.info.surface };
        flightNote = typeof msg.wings_note === "string" ? `flight (not validated): ${msg.wings_note}; shown flapping at ${WING_FLAP_HZ} Hz so it is visible (the real wingbeat is ~200 Hz)` : "";
        refreshCaption();
        const wo = (msg.world as { objects?: unknown } | undefined)?.objects;
        if (Array.isArray(wo)) {
          const sig = JSON.stringify(wo);
          if (sig !== lastObjects) {
            lastObjects = sig;
            // server frame (x forward, y left, mm; pos_mm = [x, y, height of the centre]) -> viewer (X = y, Z = x, Y above floor)
            const list = wo.flatMap((o: { id?: unknown; kind?: unknown; pos_mm?: number[]; size_mm?: unknown }) =>
              Array.isArray(o?.pos_mm) && o.pos_mm.length === 3 ? [{ id: o.id, kind: o.kind, pos_mm: objectToViewer(o.pos_mm as [number, number, number]), size_mm: o.size_mm }] : []);
            guard("objects", () => deps.sinks.setObjects?.(list));
          }
        }
        return;
      }
      case "stimulus": return;
      case "error": {
        message = typeof msg.detail === "string" ? msg.detail : "the server reported an error";
        notify();
        return;
      }
      case "stopped": { stopped = true; return; }
      default: return malformed();
    }
  };

  const onClose = (code: number, reason: string) => {
    const was = state;
    socket = null; // already closed: nothing to close again
    if (stopped && code === 1000) { finish("idle", closeMessage(1000, reason, was)); return; }
    finish("closed", closeMessage(code, reason, was));
  };

  return {
    get status() { return status(); },
    start(opts) {
      if (state !== "idle" && state !== "closed") return false;
      reset();
      let s: SocketLike;
      try { s = deps.openSocket(deps.url); } catch (e) {
        finish("closed", `could not open the live connection: ${(e as Error).message}`);
        return false;
      }
      socket = s;
      const mine = ++generation;
      s.onopen = () => {
        if (mine !== generation) return;
        const cmd: Record<string, unknown> = { type: "start", preset: opts.preset, rate_hz: opts.rateHz };
        if (opts.seed !== undefined) cmd.seed = opts.seed;
        if (opts.chunkMs !== undefined) cmd.chunk_ms = opts.chunkMs;
        if (opts.vision) cmd.vision = opts.vision;
        setState("starting");
        send(cmd);
      };
      s.onmessage = (ev) => { if (mine === generation) onMessage(ev.data); };
      s.onclose = (ev) => { if (mine === generation) onClose(ev.code, ev.reason); };
      s.onerror = () => { /* the close event that follows carries the code */ };
      setState("connecting");
      return true;
    },
    stimulus(preset, rateHz) {
      if (state !== "live") return false;
      return send({ type: "stimulus", preset, rate_hz: rateHz });
    },
    stop() {
      if (state === "idle") return;
      if (state === "closed") { finish("idle", null); return; }
      const s = socket;
      if (state === "live") send({ type: "stop" });
      finish("idle", null);
      try { s?.close(1000, "stop"); } catch { /* already gone */ }
    },
    tick(nowMs) {
      if (state !== "live") return false;
      const dt = lastTick === null ? 0 : Math.min(Math.max(nowMs - lastTick, 0), deps.maxDtMs ?? MAX_DT_MS);
      lastTick = nowMs;
      const f = playback.advance(dt);
      if (!f) return true;
      const { clip, seq } = f.item;
      if (seq === shownSeq && f.tSec === shownT && !wingsAt(clip, f.tSec)) return true; // held: nothing moved (flying wings keep flapping)
      if (seq !== shownSeq) n.played++;
      shownSeq = seq; shownT = f.tSec;
      guard("frame", () => deps.sinks.setFrame(clipMatrices(clip, f.tSec)));
      guard("world", () => deps.sinks.setWorld(worldPose(clip, f.tSec)));
      // the wingbeat is drawn on the display clock (WING_FLAP_HZ), not on simulated time: at the playback rate (and with whole
      // chunks skipped) a ~10-200 Hz beat aliases to a still wing. The recorded per-frame wings still decide whether it flies.
      const wing = wingsAt(clip, f.tSec) ? (wingCycleAt(clip, (nowMs / 1000) * WING_FLAP_HZ) ?? wingsAt(clip, f.tSec)) : null;
      if (wing || sentWings) { guard("pose", () => deps.sinks.setPose(wing ? flightWingPose(wing) : {})); sentEscape = null; }
      sentWings = wing !== null;
      const esc = wing ? false : escapeAt(clip, f.tSec);
      if (!wing && (esc !== (sentEscape ?? false) || sentEscape === null && esc)) {
        guard("pose", () => deps.sinks.setPose(escapePose(esc)));
      }
      sentEscape = esc;
      return true;
    },
    dispose() {
      const s = socket;
      release();
      playback.reset();
      state = "idle";
      try { s?.close(1000, "dispose"); } catch { /* already gone */ }
    },
  };
}
