/** DOM + scene glue for the activity wave. All the logic it uses lives in activity.ts (tested). */
import {
  binAt, createPlayback, fillBrightness, formatTime, scrub, summedBrightness, tick, togglePlay,
  type Activity, type Playback,
} from "./activity";
import type { BrainScene } from "./scene";

export interface Player {
  /** Show a recording: animate it, or draw it as one still frame when motion is reduced. */
  load(a: Activity): void;
  /** Remove the wave and the timeline. */
  clear(): void;
}

export interface PlayerOptions {
  scene: BrainScene;
  root: HTMLElement;
  reducedMotion: boolean;
  speed: number;
}

export function createPlayer({ scene, root, reducedMotion, speed }: PlayerOptions): Player {
  const playBtn = root.querySelector<HTMLButtonElement>("#tl-play")!;
  const bar = root.querySelector<HTMLInputElement>("#tl-bar")!;
  const label = root.querySelector<HTMLElement>("#tl-time")!;
  let activity: Activity | null = null;
  let pb: Playback | null = null;
  let values = new Float32Array(0);
  let raf = 0;
  let last = 0;

  const draw = () => {
    if (!activity || !pb) return;
    fillBrightness(activity, binAt(pb.timeMs, activity.binMs, activity.nBins), values);
    scene.setActivityFrame(values);
    label.textContent = formatTime(pb.timeMs);
    bar.value = String(Math.round((pb.timeMs / pb.durationMs) * Number(bar.max)));
    playBtn.textContent = pb.playing ? "pause" : "play";
    playBtn.setAttribute("aria-label", pb.playing ? "pause the activity" : "play the activity");
  };

  const frame = (now: number) => {
    raf = 0;
    if (!pb) return;
    pb = tick(pb, now - last);
    last = now;
    draw();
    if (pb.playing) raf = requestAnimationFrame(frame);
  };
  const run = () => {
    if (raf || !pb?.playing) return;
    last = performance.now();
    raf = requestAnimationFrame(frame);
  };

  playBtn.addEventListener("click", () => {
    if (!pb) return;
    pb = togglePlay(pb);
    draw();
    run();
  });
  bar.addEventListener("input", () => {
    if (!pb) return;
    pb = scrub(pb, Number(bar.value) / Number(bar.max));
    draw();
  });

  return {
    load(a) {
      cancelAnimationFrame(raf);
      raf = 0;
      activity = a;
      values = new Float32Array(a.nRows);
      scene.setActivity(a.rows);
      root.hidden = false;
      document.body.classList.add("has-activity");
      root.classList.toggle("still", reducedMotion);
      if (reducedMotion) {
        pb = null;
        summedBrightness(a, values);
        scene.setActivityFrame(values);
        label.textContent = `summed over ${Math.round(a.durationMs)} ms`;
        return;
      }
      bar.max = String(a.nBins);
      pb = createPlayback(a.durationMs, { speed });
      draw();
      run();
    },
    clear() {
      cancelAnimationFrame(raf);
      raf = 0;
      activity = null;
      pb = null;
      scene.setActivity(null);
      root.hidden = true;
      document.body.classList.remove("has-activity");
    },
  };
}
