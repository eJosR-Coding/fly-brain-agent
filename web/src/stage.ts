/** What the legend line shows (the "stage"): one value with one owner, instead of flags plus DOM text. Pure.
 *  The scene layers (highlight, cables, wave) stay separate and independent; this only says what the legend says. */
import { legendView } from "./status";

export type Stage =
  | { kind: "empty" }
  | { kind: "selection"; total: number | null; msg?: string; lit?: boolean }
  | { kind: "wave"; text: string }
  | { kind: "path"; text: string }
  | { kind: "message"; text: string };

export const EMPTY: Stage = { kind: "empty" };

export function legendOf(s: Stage): { text: string; active: boolean } {
  switch (s.kind) {
    case "empty":
      return legendView(null);
    case "selection":
      return s.lit && s.msg ? { text: s.msg, active: true } : legendView(s.total, s.msg);
    case "wave":
    case "path":
      return { text: s.text, active: true };
    case "message":
      return { text: s.text, active: false };
  }
}

/** The player changed mode. A mode legend is the wave's; "keep the wave's" repaints it only while a wave is on stage. */
export function modeStage(current: Stage, legend: string | null, base: string): Stage {
  if (legend !== null) return { kind: "wave", text: legend };
  return current.kind === "wave" && base ? { kind: "wave", text: base } : current;
}

/** A new chat clears the cables' summary, never the legend of a recording or a selection. */
export function afterNewChat(current: Stage): Stage {
  return current.kind === "path" ? EMPTY : current;
}

export interface StageLease {
  /** Paint while this lease still owns the stage (the first paint always does); false when someone took over. */
  paint(s: Stage): boolean;
  /** Give the stage back: restores the previous one only if this lease still owns it. */
  release(): boolean;
}

export class StageHolder {
  current: Stage = EMPTY;

  set(s: Stage): void {
    this.current = s;
  }

  /** For a timer that temporarily owns the legend (progress text) and must neither overwrite nor wipe a newer one. */
  lease(): StageLease {
    const previous = this.current;
    let mine: Stage | null = null;
    return {
      paint: (s) => {
        if (mine !== null && this.current !== mine) return false;
        mine = s;
        this.current = s;
        return true;
      },
      release: () => {
        if (mine === null || this.current !== mine) return false;
        mine = null;
        this.current = previous;
        return true;
      },
    };
  }
}
