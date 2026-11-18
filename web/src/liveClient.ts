/** The browser side of a live session: the real WebSocket, requestAnimationFrame and the body, around liveSession.
 *  Thin on purpose: everything with logic lives in liveSession/livePlayback/liveFrames (tested with fakes). The draw loop
 *  runs only while a session is live; the glass-cube sandbox is open for as long as a session runs.
 *
 *  DEV ENTRY: the "live" chip in main.ts and `window.flyLive.start(preset, rate)` call this until the agent tools of the
 *  playground (task P4) exist; P4 replaces them with chat-driven control (no buttons). */
import type { WorldPose } from "./bodyClip";
import type { JointId } from "./bodyPose";
import { createLiveSession, type LiveSession, type LiveStatus, type SocketLike } from "./liveSession";

/** The dev entry's drive: forward-walking neurons DNg100 at 50 Hz (the same preset as the "walk (DNg100)" chip). */
export const DEV_PRESET = "dng100";
export const DEV_RATE_HZ = 50;

/** The API's live socket, on the page's own host (the dev server proxies /api with WebSocket upgrades). */
export function liveUrl(loc: { protocol: string; host: string }): string {
  return `${loc.protocol === "https:" ? "wss" : "ws"}://${loc.host}/api/ws/live`;
}

export interface LiveBody {
  setFrame(mats: ReadonlyMap<string, ArrayLike<number>> | null): void;
  setWorld(w: WorldPose | null): void;
  setPose(angles: Partial<Record<JointId, number>>): void;
  setCaption(text: string): void;
  setSandbox(on: boolean): void;
}

export interface LiveClientDeps {
  body: LiveBody;
  defaultCaption: string;
  location?: { protocol: string; host: string };
  openSocket?: (url: string) => SocketLike;
  schedule?: (cb: (t: number) => void) => number;
  cancel?: (id: number) => void;
  every?: (fn: () => void, ms: number) => () => void;
  /** Called before a session starts: hand the stage over (stop other body players). */
  prepare?: () => void;
}

export interface LiveClient {
  readonly status: LiveStatus;
  start(preset: string | null, rateHz: number): void;
  stop(): void;
  /** Start the dev preset, or stop a running session. */
  toggle(): void;
  stimulus(preset: string | null, rateHz: number): boolean;
  onChange(fn: (s: LiveStatus) => void): () => void;
  dispose(): void;
}

export function createLiveClient(deps: LiveClientDeps): LiveClient {
  const schedule = deps.schedule ?? ((cb) => requestAnimationFrame(cb));
  const cancel = deps.cancel ?? ((id) => cancelAnimationFrame(id));
  const listeners = new Set<(s: LiveStatus) => void>();
  let frameId: number | null = null;
  let sandbox = false;
  let session: LiveSession;

  const loop = (t: number) => {
    frameId = null;
    if (session.tick(t)) frameId = schedule(loop);
  };
  const onChange = (s: LiveStatus) => {
    const running = s.state === "connecting" || s.state === "starting" || s.state === "live";
    if (running !== sandbox) { sandbox = running; deps.body.setSandbox(running); }
    if (s.state === "live" && frameId === null) frameId = schedule(loop);
    if (s.state !== "live" && frameId !== null) { cancel(frameId); frameId = null; }
    for (const fn of listeners) fn(s);
  };
  session = createLiveSession({
    url: liveUrl(deps.location ?? window.location),
    openSocket: deps.openSocket ?? ((url) => new WebSocket(url) as unknown as SocketLike),
    sinks: deps.body,
    every: deps.every ?? ((fn, ms) => { const id = setInterval(fn, ms); return () => clearInterval(id); }),
    defaultCaption: deps.defaultCaption,
    onChange,
  });
  const busy = () => session.status.state !== "idle" && session.status.state !== "closed";

  return {
    get status() { return session.status; },
    start(preset, rateHz) {
      if (busy()) return;
      deps.prepare?.();
      session.start({ preset, rateHz });
    },
    stop() { session.stop(); },
    toggle() { if (busy()) session.stop(); else this.start(DEV_PRESET, DEV_RATE_HZ); },
    stimulus: (preset, rateHz) => session.stimulus(preset, rateHz),
    onChange(fn) { listeners.add(fn); return () => listeners.delete(fn); },
    dispose() {
      session.dispose();
      if (frameId !== null) { cancel(frameId); frameId = null; }
      listeners.clear();
    },
  };
}
