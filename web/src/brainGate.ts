/** All-or-nothing gate for the real brain scene: idle -> loading -> ready | failed.
 *  Pure (no DOM). `ensure()` is idempotent: concurrent callers share ONE load, a ready gate stays ready
 *  (no unloading) and a failed gate retries on the next `ensure()`. */

export type GateState = "idle" | "loading" | "ready" | "failed";

export interface BrainGate<T> {
  ensure(): Promise<T>;
  state(): GateState;
  /** Called on every state change; returns the unsubscribe function. */
  subscribe(cb: (s: GateState) => void): () => void;
}

export function createBrainGate<T>(opts: { load: () => Promise<T> }): BrainGate<T> {
  let state: GateState = "idle";
  let value: T;
  let inflight: Promise<T> | null = null;
  const listeners = new Set<(s: GateState) => void>();
  const set = (s: GateState) => {
    state = s;
    for (const cb of [...listeners]) {
      try {
        cb(s);
      } catch {
        // a broken listener must not break the load
      }
    }
  };
  return {
    ensure() {
      if (state === "ready") return Promise.resolve(value);
      if (inflight) return inflight;
      set("loading");
      inflight = (async () => {
        try {
          value = await opts.load();
          set("ready");
          return value;
        } catch (e) {
          set("failed");
          throw e;
        } finally {
          inflight = null;
        }
      })();
      return inflight;
    },
    state: () => state,
    subscribe(cb) {
      listeners.add(cb);
      return () => void listeners.delete(cb);
    },
  };
}

/** Zoom-in intent on the preview: a wheel that scrolls forward, or a trackpad pinch (ctrl + wheel). */
export function isZoomIntent(e: { deltaY: number; ctrlKey: boolean }): boolean {
  return e.deltaY < 0 || (e.ctrlKey && e.deltaY !== 0);
}

export function isActivationKey(key: string): boolean {
  return key === "Enter" || key === " " || key === "Spacebar";
}
