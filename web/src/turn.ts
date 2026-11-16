/** Pure lifecycle of one chat turn. Resources (interval, AbortController) stay with the DOM glue; only the state lives here. */
import type { ChatHandlers } from "./chat";

export type Turn =
  | { k: "idle" }
  | { k: "pending"; id: number } // sent, no event yet
  | { k: "streaming"; id: number } // events are arriving
  | { k: "stopping"; id: number }; // the user pressed stop; waiting for the stream to unwind

export type TurnEvent =
  | { type: "start"; id: number }
  | { type: "event" }
  | { type: "stop" }
  | { type: "done"; id: number };

export const IDLE: Turn = { k: "idle" };

export const isBusy = (t: Turn): boolean => t.k !== "idle";

/** Next state; an invalid transition returns the very same state object. */
export function turnStep(t: Turn, e: TurnEvent): Turn {
  switch (e.type) {
    case "start": return t.k === "idle" ? { k: "pending", id: e.id } : t;
    case "event": return t.k === "pending" ? { k: "streaming", id: t.id } : t;
    case "stop": return t.k === "pending" || t.k === "streaming" ? { k: "stopping", id: t.id } : t;
    case "done": return t.k !== "idle" && t.id === e.id ? IDLE : t;
  }
}

const ALWAYS: ReadonlySet<keyof ChatHandlers> = new Set(["onDone"]);
const WHILE_STOPPING: ReadonlySet<keyof ChatHandlers> = new Set(["onError", "onDone"]);

/**
 * Wrap the handlers of turn `id`: an event reaches them only while that turn is current (not idle, not replaced),
 * and a stopping turn only reports its error and its end. `onDone` always passes (streamChat fires it exactly once).
 * A passing event also advances pending -> streaming through `onEvent`.
 */
export function guardHandlers(h: ChatHandlers, id: number, state: () => Turn, onEvent: () => void): ChatHandlers {
  const out: Record<string, unknown> = {};
  for (const key of Object.keys(h) as (keyof ChatHandlers)[]) {
    const fn = h[key] as ((...a: unknown[]) => unknown) | undefined;
    if (typeof fn !== "function") continue;
    out[key] = (...args: unknown[]) => {
      if (!ALWAYS.has(key)) {
        const s = state();
        if (s.k === "idle" || s.id !== id) return;
        if (s.k === "stopping" && !WHILE_STOPPING.has(key)) return;
        onEvent();
      }
      return fn(...args);
    };
  }
  return out as unknown as ChatHandlers;
}
