/** A body clip driven by the brain recording's playback clock (not by its own timer), so the activity and the body move
 *  together. Pure: the sinks are injected. The clip is a physics run from OUR mapping of the recording's descending
 *  neurons; the flow that supplies it carries the caption that says so. */
import { clipMatrices, escapeAt, worldPose, type WalkClip, type WorldPose } from "./bodyClip";
import type { JointId } from "./bodyPose";
import { escapePose } from "./bodyWorld";

type Mats = ReadonlyMap<string, ArrayLike<number>>;
type Pose = Partial<Record<JointId, number>>;

export interface ClipPlayerDeps {
  setFrame: (mats: Mats | null) => void;
  setWorld: (w: WorldPose | null) => void;
  setPose: (angles: Pose) => void;
  /** show the final frame at once and ignore the clock */
  reducedMotion?: boolean;
}

export interface ClipPlayer {
  /** True while a clip is set. */
  readonly active: boolean;
  /** Replace the clip (shown at once under reduced motion; otherwise the next clock tick shows it). */
  setClip(clip: WalkClip): void;
  /** The playback clock in ms, as the player emits it; null = the recording is gone: rest. Ignored without a clip. */
  onTime(ms: number | null): void;
  /** Drop the clip and return the body to rest. Quiet when there is none. */
  clear(): void;
}

export function createClipPlayer(deps: ClipPlayerDeps): ClipPlayer {
  let clip: WalkClip | null = null;
  let sentEscape: boolean | null = null;
  const guard = (what: string, fn: () => void) => {
    try { fn(); } catch (e) { console.error(`body clip ${what} failed`, e); }
  };
  const rest = () => {
    guard("frame", () => deps.setFrame(null));
    guard("world", () => deps.setWorld(null));
    if (sentEscape !== false) guard("pose", () => deps.setPose({}));
    sentEscape = false;
  };
  const show = (c: WalkClip, t: number) => {
    guard("frame", () => deps.setFrame(clipMatrices(c, t)));
    guard("world", () => deps.setWorld(worldPose(c, t)));
    const esc = escapeAt(c, t);
    if (esc !== sentEscape) {
      guard("pose", () => deps.setPose(escapePose(esc)));
      sentEscape = esc;
    }
  };
  return {
    get active() { return clip !== null; },
    setClip(c) {
      clip = c;
      sentEscape = null;
      if (deps.reducedMotion) show(c, c.durationS); // loop:false holds the last frame, whatever t is
    },
    onTime(ms) {
      if (!clip || deps.reducedMotion) return;
      if (ms === null || !Number.isFinite(ms)) { rest(); return; }
      show(clip, ms / 1000);
    },
    clear() {
      if (!clip) return;
      clip = null;
      rest();
    },
  };
}
