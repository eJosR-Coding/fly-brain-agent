/**
 * Pure glue between the playback clock and the body pose. No DOM, no three.js.
 *
 * Each clock tick samples every named readout of the loaded activity, maps the rates to joint angles with `poseFor`
 * and hands the pose to `setPose`. Honesty: the angles come from `bodyPose.ts`; this module adds no motion of its own.
 */
import type { NamedReadout } from "./activity";
import { MAPPING, poseFor, sampleReadout, type JointId } from "./bodyPose";

type Pose = Partial<Record<JointId, number>>;

export interface DrivenActivity { readouts?: NamedReadout[]; binMs: number }

export interface BodyDriverOptions {
  setPose: (angles: Pose) => void;
  /** passed to `sampleReadout`; default 1 (no smoothing) */
  smoothBins?: number;
}

export interface BodyDriver {
  /** Replace the activity; null returns to rest at once. */
  setActivity(act: DrivenActivity | null): void;
  /** Playback time in ms; null or non-finite means rest. */
  onTime(ms: number | null): void;
  /** After this nothing is ever sent. */
  dispose(): void;
}

const isMapped = (name: string): boolean => Object.prototype.hasOwnProperty.call(MAPPING, name);

/** Names of the readouts that map to the body (in `MAPPING`) and carry data, each once, in activity order. */
export function activeReadoutNames(act: DrivenActivity | null): string[] {
  const out: string[] = [];
  for (const r of act?.readouts ?? []) {
    if (r.name !== null && isMapped(r.name) && r.hz.length > 0 && !out.includes(r.name)) out.push(r.name);
  }
  return out;
}

function samePose(a: Pose, b: Pose): boolean {
  const ka = Object.keys(a) as JointId[];
  if (ka.length !== Object.keys(b).length) return false;
  return ka.every((k) => k in b && a[k] === b[k]);
}

export function createBodyDriver(opts: BodyDriverOptions): BodyDriver {
  let activity: DrivenActivity | null = null;
  let disposed = false;
  /** the last pose the host accepted; the body starts at rest */
  let sent: Pose = {};
  const logged = new Set<string>();

  function send(pose: Pose): void {
    if (disposed || samePose(sent, pose)) return;
    try {
      opts.setPose(pose);
      sent = pose;
    } catch (e) {
      // keep `sent` as it was: the failed pose is retried on the next tick
      const msg = e instanceof Error ? e.message : String(e);
      if (!logged.has(msg)) {
        logged.add(msg);
        console.error("bodyDriver: setPose failed:", e);
      }
    }
  }

  function poseAt(ms: number | null): Pose {
    if (!activity || ms === null || !Number.isFinite(ms)) return {};
    if (!(activity.binMs > 0) || !Number.isFinite(activity.binMs)) return {};
    const hz: Record<string, number> = {};
    for (const r of activity.readouts ?? []) {
      if (r.name === null || r.hz.length === 0) continue;
      hz[r.name] = sampleReadout(r.hz, activity.binMs, ms, opts.smoothBins);
    }
    return poseFor(hz, ms);
  }

  return {
    setActivity(act) {
      if (disposed) return;
      activity = act;
      if (act === null) send({});
    },
    onTime(ms) {
      if (disposed) return;
      send(poseAt(ms));
    },
    dispose() {
      disposed = true;
      activity = null;
    },
  };
}
