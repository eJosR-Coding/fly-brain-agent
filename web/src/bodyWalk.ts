/** Playback of the physics walking clip on the body. Pure logic: the clip loader, the frame sink, the caption sink
 *  and the frame scheduler are all injected, so it runs under test with fakes (no DOM, no three, no timers).
 *
 *  Honesty: the clip is a published locomotion controller (flygym's tripod CPG walker) replayed on our mesh. The brain
 *  model does not drive it, and the caption says so. */
import { clipMatrices, type WalkClip } from "./bodyClip";

/** Shown under the body while the clip plays. Says what it is and what it is not. */
export const WALK_CAPTION = "NeuroMechFly v2 physics (flygym 2.1.1): tripod walking controller, not yet driven by the brain";

/** Default playback slowdown: 1 = real time. A real fly steps about 12 times a second (one step every ~83 ms); 20x
 *  slow motion was tried and found too slow by the user, so the demo plays at the real speed. A slowdown above 1 is
 *  still supported and is then stated in the caption. */
export const SLOW_MOTION = 1;

/** Usable slowdown: a finite number above 0, else real time (1). */
const slowdownOf = (s: number | undefined): number => (s !== undefined && Number.isFinite(s) && s > 0 ? s : 1);

/** The caption for a slowdown: the honest base, plus the slowdown when it is not real time. */
export function walkCaption(slowdown: number): string {
  const s = slowdownOf(slowdown);
  return s === 1 ? WALK_CAPTION : `${WALK_CAPTION}; slow motion x${s} (a real fly steps about 12 times a second)`;
}

/** One static frame for reduced motion: the middle of the first run of frames with exactly three legs in stance
 *  (adhesion on, all the same legs), i.e. a mid-stance tripod. Frame 0 when the clip carries no usable flags. */
export function pickStaticFrame(clip: WalkClip): number {
  const nLegs = clip.legs.length;
  if (nLegs === 0 || clip.adhesion.length !== clip.nFrames * nLegs) return 0;
  const pattern = (f: number) => clip.adhesion.subarray(f * nLegs, (f + 1) * nLegs).join("");
  const stance = (f: number) => clip.adhesion.subarray(f * nLegs, (f + 1) * nLegs).reduce((n, v) => n + v, 0) === 3;
  for (let start = 0; start < clip.nFrames; start++) {
    if (!stance(start)) continue;
    let end = start;
    while (end + 1 < clip.nFrames && pattern(end + 1) === pattern(start)) end++;
    return start + Math.floor((end - start + 1) / 2);
  }
  return 0;
}

export interface WalkPlayerDeps {
  load: () => Promise<WalkClip>;
  /** Frame sink: per-body matrices, or null for rest. */
  setFrame: (mats: ReadonlyMap<string, ArrayLike<number>> | null) => void;
  setCaption: (text: string) => void;
  /** Caption restored on stop. */
  defaultCaption: string;
  /** requestAnimationFrame-like: the callback receives a timestamp in ms. Only scheduled while playing. */
  schedule: (cb: (timestampMs: number) => void) => number;
  cancel: (id: number) => void;
  /** One static frame instead of motion. */
  reducedMotion?: boolean;
  /** Called with the new `playing` value whenever it changes (for the chip's pressed state). */
  onChange?: (playing: boolean) => void;
  /** Wall-clock time per clip time (default SLOW_MOTION; 1 = real time). Non-finite or <= 0 means real time. */
  slowdown?: number;
}

export interface WalkPlayer {
  readonly playing: boolean;
  start(): Promise<void>;
  /** Stops and returns the body to rest. A no-op when nothing is playing. */
  stop(): void;
  toggle(): Promise<void>;
  /** Stops (quietly if idle) and ignores every later call. */
  dispose(): void;
}

export function createWalkPlayer(deps: WalkPlayerDeps): WalkPlayer {
  let playing = false;
  let disposed = false;
  let generation = 0; // bumped on every stop: a start that finishes loading after a stop is dropped
  let clip: WalkClip | null = null;
  let starting: Promise<void> | null = null;
  let frameId: number | null = null;
  const slowdown = slowdownOf("slowdown" in deps ? deps.slowdown : SLOW_MOTION);

  const setPlaying = (v: boolean) => {
    if (playing === v) return;
    playing = v;
    deps.onChange?.(v);
  };
  const toRest = () => {
    generation++;
    if (frameId !== null) { deps.cancel(frameId); frameId = null; }
    setPlaying(false);
    deps.setFrame(null);
    deps.setCaption(deps.defaultCaption);
  };

  const run = async (): Promise<void> => {
    const mine = ++generation;
    setPlaying(true);
    try {
      clip ??= await deps.load();
    } catch (e) {
      if (mine !== generation) return; // stopped meanwhile: already at rest
      console.warn("physics walk not shown:", e);
      toRest();
      return;
    }
    if (mine !== generation || disposed) return;
    const c = clip;
    if (deps.reducedMotion) {
      deps.setCaption(WALK_CAPTION); // one static frame: no playback speed to report
      deps.setFrame(clipMatrices(c, pickStaticFrame(c) / c.sampleRateHz));
      return;
    }
    deps.setCaption(walkCaption(slowdown));
    let t0: number | null = null;
    const frame = (ts: number) => {
      frameId = null;
      if (mine !== generation) return;
      t0 ??= ts;
      deps.setFrame(clipMatrices(c, (ts - t0) / 1000 / slowdown));
      frameId = deps.schedule(frame);
    };
    frameId = deps.schedule(frame);
  };

  return {
    get playing() { return playing; },
    start() {
      if (disposed) return Promise.resolve();
      if (playing) return starting ?? Promise.resolve();
      starting = run().finally(() => { starting = null; });
      return starting;
    },
    stop() {
      if (playing) toRest();
    },
    async toggle() {
      if (playing) this.stop();
      else await this.start();
    },
    dispose() {
      if (playing) toRest();
      disposed = true;
    },
  };
}
