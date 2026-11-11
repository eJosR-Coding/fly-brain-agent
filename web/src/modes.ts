/** The ways to color the activity layer. `diff` needs a baseline recording to compare against. */
import { diffLegend } from "./diff";

export type ViewMode = "wave" | "latency" | "diff";

export function modesFor(hasCompare: boolean): ViewMode[] {
  return hasCompare ? ["wave", "latency", "diff"] : ["wave", "latency"];
}

/** Keep the current mode when it is still offered, else go back to the wave. */
export function fallbackMode(current: ViewMode, hasCompare: boolean): ViewMode {
  return modesFor(hasCompare).includes(current) ? current : "wave";
}

export interface ViewPlan {
  /** the mode actually shown (a diff request without a baseline becomes the wave) */
  mode: ViewMode;
  /** which rows the scene lights: the recorded ones or the union of both runs */
  rows: "activity" | "diff";
  /** neurons get ring/dot/cross shapes (the diff map) */
  shapes: boolean;
  /** the wave runs frame by frame */
  animate: boolean;
  /** the key under the timeline */
  scale: "none" | "latency" | "diffKey";
  /** the timeline controls are hidden: a static map has no playhead */
  staticView: boolean;
  /** text of the time label: the playhead, the summed still, or nothing */
  timeLabel: "playhead" | "summed" | "none";
}

/** Everything the player must do for a requested mode, as data (so it is testable without a DOM). */
export function viewPlan(requested: ViewMode, ctx: { hasDiff: boolean; reducedMotion: boolean }): ViewPlan {
  const mode = fallbackMode(requested, ctx.hasDiff);
  if (mode === "wave") {
    return {
      mode, rows: "activity", shapes: false, animate: !ctx.reducedMotion, scale: "none", staticView: false,
      timeLabel: ctx.reducedMotion ? "summed" : "playhead",
    };
  }
  if (mode === "diff") {
    return { mode, rows: "diff", shapes: true, animate: false, scale: "diffKey", staticView: true, timeLabel: "none" };
  }
  return { mode, rows: "activity", shapes: false, animate: false, scale: "latency", staticView: true, timeLabel: "none" };
}

/** Legend text for a static mode; null keeps the wave's own legend. */
export function modeLegend(
  mode: ViewMode,
  info: { nRows: number; diff: { lost: number; gained: number } | null; readout: { label: string; before: number; after: number } | null },
): string | null {
  if (mode === "latency") return `latency · ${info.nRows.toLocaleString("en-US")} neurons spiked`;
  if (mode === "diff" && info.diff) return diffLegend(info.diff.lost, info.diff.gained, info.readout);
  return null;
}
