/**
 * Pure glue between a loaded recording, the body driver and the body caption. No DOM, no three.js, no timers.
 *
 * The body moves only while a recording is loaded and the playback clock ticks; the caption says so honestly
 * (it is built from the mapping table in bodyPose.ts, never written by hand here).
 */
import type { NamedReadout } from "./activity";
import { activeReadoutNames, type BodyDriver } from "./bodyDriver";
import { captionFor } from "./bodyPose";

/** Shown when no recording is driving the body. Credits the mesh and says the mapping is ours. */
export const DEFAULT_BODY_CAPTION =
  "NeuroMechFly v2 micro-CT mesh (Apache-2.0); it moves only when a simulation runs, with a mapping chosen by us";

export interface AnimatedActivity { readouts?: NamedReadout[]; binMs: number; durationMs: number }

export interface BodyAnimationOptions {
  driver: Pick<BodyDriver, "setActivity" | "onTime" | "dispose">;
  setCaption: (text: string) => void;
  defaultCaption: string;
  /** static final pose instead of a moving one */
  reducedMotion?: boolean;
}

export interface BodyAnimation {
  /** The MAIN recording now on stage (never a baseline), or null when it is cleared. */
  setActivity(act: AnimatedActivity | null): void;
  /** Subscribe this to the player clock: it receives exactly what the player emits. */
  onTime(ms: number | null): void;
  /** Rest, default caption, and nothing is ever sent again. */
  dispose(): void;
}

export function createBodyAnimation({ driver, setCaption, defaultCaption, reducedMotion = false }: BodyAnimationOptions): BodyAnimation {
  let activity: AnimatedActivity | null = null;
  let disposed = false;
  return {
    setActivity(act) {
      if (disposed) return;
      activity = act;
      driver.setActivity(act === null ? null : { readouts: act.readouts, binMs: act.binMs });
      const names = activeReadoutNames(act);
      setCaption(names.length > 0 ? captionFor(names) : defaultCaption);
      // reduced motion: no clock ticks, so pose the final frame right away (the player's own signal arrives too, and is the same)
      if (reducedMotion && act !== null) driver.onTime(act.durationMs);
    },
    onTime(ms) {
      if (disposed) return;
      if (reducedMotion) {
        // the player emits null on load under reduced motion (no playhead): keep the static final pose while a recording is loaded
        driver.onTime(activity ? activity.durationMs : null);
        return;
      }
      // The player emits the full duration when playback finishes and keeps the playhead there: a body held in the
      // last bin's pose forever would be a lie, so reaching the end means rest.
      if (ms !== null && activity && ms >= activity.durationMs) {
        driver.onTime(null);
        return;
      }
      driver.onTime(ms);
    },
    dispose() {
      if (disposed) return;
      driver.onTime(null);
      driver.setActivity(null);
      setCaption(defaultCaption);
      driver.dispose();
      disposed = true;
      activity = null;
    },
  };
}
