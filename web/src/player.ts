/** DOM + scene glue for the activity wave. All the logic it uses lives in activity.ts (tested). */
import {
  binAt, createPlayback, fillBrightness, formatTime, playLabel, scrub, summedBrightness, tick, togglePlay,
  type Activity, type Playback,
} from "./activity";
import { computeDiff, diffBrightness, diffLegend, diffShapes, meanHz, type Diff } from "./diff";
import { fillLatency, latencyGradientStops, latencyTicks } from "./latency";
import { fallbackMode, modesFor, type ViewMode } from "./modes";
import type { BrainScene } from "./scene";

export interface Player {
  /** Show a recording: animate it, or draw it as one still frame when motion is reduced. */
  load(a: Activity, baseline?: Activity | null): void;
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
  let diff: Diff | null = null;
  let diffValues = new Float32Array(0);
  let diffReadout: { label: string; before: number; after: number } | null = null;
  let pb: Playback | null = null;
  let values = new Float32Array(0);
  let shapeValues = new Float32Array(0);
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
    const hasCompare = diff !== null;
    modesEl.replaceChildren(
      ...modesFor(hasCompare).map((m) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "chip mode-btn";
        b.textContent = m;
        b.setAttribute("aria-pressed", String(m === mode));
        b.title = m === "wave" ? "spikes over time"
          : m === "latency" ? "when each neuron fired first (static map)"
          : "what the silencing lost (rings) and gained (dots) compared with the baseline run";
        b.addEventListener("click", () => api.setMode(m));
        return b;
      }),
    );
    renderScale(a);
  };

  const renderDiffKey = () => {
    const item = (cls: string, text: string, glyph = "") => {
      const el = document.createElement("span");
      el.className = "k-item";
      const mark = document.createElement("i");
      mark.className = cls;
      mark.textContent = glyph;
      el.append(mark, text);
      return el;
    };
    const key = document.createElement("div");
    key.className = "diff-key";
    key.append(item("k-ring", "lost"), item("k-dot", "gained"), item("k-cross", "silenced", "×"));
    scaleEl.replaceChildren(key);
  };

  const applyMode = () => {
    if (!activity) return;
    const wave = mode === "wave";
    root.classList.toggle("static", !wave);
    scaleEl.hidden = wave;
    if (mode === "latency") renderScale(activity);
    else if (mode === "diff") renderDiffKey();
    for (const b of modesEl.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.textContent === mode));
    cancelAnimationFrame(raf);
    raf = 0;
    if (wave) {
      scene.setActivity(activity.rows);
      scene.setActivityShapes(null);
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
    if (mode === "diff" && diff) {
      scene.setActivity(diff.rows);
      diffBrightness(diff, diffValues);
      scene.setActivityFrame(diffValues);
      diffShapes(diff, shapeValues);
      scene.setActivityShapes(shapeValues);
    } else {
      scene.setActivity(activity.rows); // back from the diff: the recorded rows, all dots
      fillLatency(activity, values);
      scene.setActivityFrame(values);
      scene.setActivityShapes(null);
    }
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
      mode = fallbackMode(next, diff !== null);
      applyMode();
      onMode?.(
        mode,
        mode === "latency" ? `latency · ${activity.nRows.toLocaleString("en-US")} neurons spiked`
          : mode === "diff" && diff ? diffLegend(diff.lost, diff.gained, diffReadout) : null,
      );
    },
    load(a, baseline = null) {
      cancelAnimationFrame(raf);
      raf = 0;
      activity = a;
      values = new Float32Array(a.nRows);
      diff = null;
      diffReadout = null;
      if (baseline) {
        try {
          diff = computeDiff(baseline, a, a.silencedRows ?? []);
          diffValues = new Float32Array(diff.rows.length);
          shapeValues = new Float32Array(diff.rows.length);
          if (a.readout && baseline.readout) {
            diffReadout = { label: a.readout.label, before: meanHz(baseline.readout.hz), after: meanHz(a.readout.hz) };
          }
        } catch {
          diff = null; // mismatched recordings: no difference map, the rest still works
        }
      }
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
