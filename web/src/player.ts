/** DOM + scene glue for the activity wave. All the logic it uses lives in activity.ts (tested). */
import {
  binAt, createPlayback, fillBrightness, formatTime, playLabel, scrub, summedBrightness, tick, togglePlay,
  type Activity, type Playback,
} from "./activity";
import { fillLatency, latencyGradientStops, latencyTicks } from "./latency";
import { fallbackMode, modesFor, type ViewMode } from "./modes";
import type { BrainScene } from "./scene";

export interface Player {
  /** Show a recording: animate it, or draw it as one still frame when motion is reduced. */
  load(a: Activity): void;
  /** Switch between the wave and the static maps (latency). */
  setMode(mode: ViewMode): void;
  readonly mode: ViewMode;
  /** Remove the wave and the timeline. */
  clear(): void;
}

export interface PlayerOptions {
  scene: BrainScene;
  root: HTMLElement;
  reducedMotion: boolean;
  speed: number;
  /** called when the shown time changes (ms), or with null when the playhead should rest/hide */
  onTime?: (ms: number | null) => void;
  /** called when the mode changes, with the text for the legend line (null: keep the wave's) */
  onMode?: (mode: ViewMode, legend: string | null) => void;
}

export function createPlayer({ scene, root, reducedMotion, speed, onTime, onMode }: PlayerOptions): Player {
  const playBtn = root.querySelector<HTMLButtonElement>("#tl-play")!;
  const bar = root.querySelector<HTMLInputElement>("#tl-bar")!;
  const label = root.querySelector<HTMLElement>("#tl-time")!;
  const modesEl = root.querySelector<HTMLElement>("#tl-modes")!;
  const scaleEl = root.querySelector<HTMLElement>("#tl-scale")!;
  let mode: ViewMode = "wave";
  let activity: Activity | null = null;
  let pb: Playback | null = null;
  let values = new Float32Array(0);
  let raf = 0;
  let last = 0;

  const draw = () => {
    if (!activity || !pb) return;
    onTime?.(pb.timeMs);
    if (!pb.playing && pb.timeMs >= pb.durationMs) summedBrightness(activity, values); // finished: everything that fired
    else fillBrightness(activity, binAt(pb.timeMs, activity.binMs, activity.nBins), values);
    scene.setActivityFrame(values);
    label.textContent = formatTime(pb.timeMs);
    bar.value = String(Math.round((pb.timeMs / pb.durationMs) * Number(bar.max)));
    const lab = playLabel(pb);
    playBtn.textContent = lab;
    playBtn.setAttribute("aria-label", `${lab} the activity`);
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

  const renderScale = (a: Activity) => {
    const bar = document.createElement("div");
    bar.className = "scale-bar";
    bar.style.background = `linear-gradient(to right, ${latencyGradientStops()
      .map((st) => `rgb(${st.grey},${st.grey},${st.grey}) ${st.frac * 100}%`).join(", ")})`;
    const ticks = document.createElement("div");
    ticks.className = "scale-ticks";
    for (const t of latencyTicks(a.durationMs)) {
      const el = document.createElement("span");
      el.textContent = t.label;
      ticks.appendChild(el);
    }
    const cap = document.createElement("div");
    cap.className = "scale-cap";
    cap.textContent = "first spike";
    scaleEl.replaceChildren(cap, bar, ticks);
  };

  const renderModes = (a: Activity) => {
    const hasCompare = false; // the difference map joins in with its own commit
    modesEl.replaceChildren(
      ...modesFor(hasCompare).map((m) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "chip mode-btn";
        b.textContent = m;
        b.setAttribute("aria-pressed", String(m === mode));
        b.title = m === "wave" ? "spikes over time" : "when each neuron fired first (static map)";
        b.addEventListener("click", () => api.setMode(m));
        return b;
      }),
    );
    renderScale(a);
  };

  const applyMode = () => {
    if (!activity) return;
    const wave = mode === "wave";
    root.classList.toggle("static", !wave);
    scaleEl.hidden = mode !== "latency";
    for (const b of modesEl.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.textContent === mode));
    cancelAnimationFrame(raf);
    raf = 0;
    if (wave) {
      if (reducedMotion || !pb) {
        summedBrightness(activity, values);
        scene.setActivityFrame(values);
        label.textContent = `summed over ${Math.round(activity.durationMs)} ms`;
        onTime?.(null);
        return;
      }
      draw();
      run();
      return;
    }
    fillLatency(activity, values);
    scene.setActivityFrame(values);
    label.textContent = "";
    onTime?.(null);
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

  const api: Player = {
    get mode() {
      return mode;
    },
    setMode(next) {
      if (!activity) return;
      mode = fallbackMode(next, false);
      applyMode();
      onMode?.(mode, mode === "latency" ? `latency · ${activity.nRows.toLocaleString("en-US")} neurons spiked` : null);
    },
    load(a) {
      cancelAnimationFrame(raf);
      raf = 0;
      activity = a;
      values = new Float32Array(a.nRows);
      scene.setActivity(a.rows);
      root.hidden = false;
      document.body.classList.add("has-activity");
      root.classList.toggle("still", reducedMotion);
      mode = "wave";
      renderModes(a);
      pb = reducedMotion ? null : createPlayback(a.durationMs, { speed });
      if (pb) bar.max = String(a.nBins);
      applyMode();
    },
    clear() {
      cancelAnimationFrame(raf);
      raf = 0;
      activity = null;
      pb = null;
      scene.setActivity(null);
      root.hidden = true;
      document.body.classList.remove("has-activity");
      onTime?.(null);
    },
  };
  return api;
}
