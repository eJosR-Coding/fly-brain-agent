/** Pure logic of the collapsible tool trace: the summary row text and the remembered open/closed state. */

export interface TraceTool {
  name: string;
  state: "running" | "done" | "failed" | "ended";
  /** final duration, null while running (or when the turn was cut off) */
  ms: number | null;
  /** performance.now() when it started */
  startedAt: number;
  /** live progress label, e.g. "screen · 3/8" */
  progress?: string;
}

const plural = (n: number) => `${n} tool${n === 1 ? "" : "s"}`;

/** "Used 3 tools · 14.2 s", "3 tools · 1 failed · 6.0 s", or "running stimulate · 6 s" while a tool runs. */
export function summarizeTools(tools: readonly TraceTool[], now: number): string {
  if (!tools.length) return "";
  const live = tools.filter((t) => t.state === "running");
  if (live.length) {
    const cur = live.reduce((a, b) => (b.startedAt >= a.startedAt ? b : a));
    const secs = Math.floor(Math.max(0, now - cur.startedAt) / 1000);
    return `running ${cur.progress ?? cur.name} · ${secs} s`;
  }
  const failed = tools.filter((t) => t.state === "failed").length;
  const total = (tools.reduce((sum, t) => sum + (t.ms ?? 0), 0) / 1000).toFixed(1);
  return failed ? `${plural(tools.length)} · ${failed} failed · ${total} s` : `Used ${plural(tools.length)} · ${total} s`;
}

export const EXPANDED_KEY = "flybrain.tools.expanded";
type Store = Pick<Storage, "getItem" | "setItem">;

function defaultStore(): Store | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null; // blocked storage can throw on access
  }
}

/** The remembered preference; collapsed unless the user opened it. */
export function loadExpanded(store: Store | null = defaultStore()): boolean {
  try {
    return store?.getItem(EXPANDED_KEY) === "1";
  } catch {
    return false;
  }
}

export function saveExpanded(expanded: boolean, store: Store | null = defaultStore()): void {
  try {
    store?.setItem(EXPANDED_KEY, expanded ? "1" : "0");
  } catch {
    /* private mode or blocked storage: the preference just is not remembered */
  }
}
