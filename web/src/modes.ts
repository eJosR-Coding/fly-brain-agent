/** The ways to color the activity layer. `diff` needs a baseline recording to compare against. */
export type ViewMode = "wave" | "latency" | "diff";

export function modesFor(hasCompare: boolean): ViewMode[] {
  return hasCompare ? ["wave", "latency", "diff"] : ["wave", "latency"];
}

/** Keep the current mode when it is still offered, else go back to the wave. */
export function fallbackMode(current: ViewMode, hasCompare: boolean): ViewMode {
  return modesFor(hasCompare).includes(current) ? current : "wave";
}
