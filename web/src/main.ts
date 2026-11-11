import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "./style.css";
import { mountChat } from "./chat";
import { parseModels } from "./models";
import { fetchIndices, loadBrain } from "./brainData";
import { createScene, type BrainScene } from "./scene";
import { Latest, runLatest } from "./latest";
import { deriveStatus, legendView, parseHealth, versionLine, type Health } from "./status";
import { readTheme } from "./theme";
import { loadLandmarks, loadRegions, type Regions } from "./regions";
import { activityLegend, formatSimulating, loadActivity, parseSimulation, type Activity } from "./activity";
import { createPlayer, type Player } from "./player";
import { mountInspector } from "./inspector";
import { pathLegend } from "./path";
import { greyHex, keyRows, loadNt, ntTier, type Transmitters } from "./transmitters";
import type { Theme } from "./theme";

const CHIPS: { label: string; params: Record<string, string> | null }[] = [
  { label: "dopamine", params: { nt: "dopamine" } },
  { label: "dopamine · left", params: { nt: "dopamine", side: "left" } },
  { label: "GABA", params: { nt: "gaba" } },
  { label: "serotonin", params: { nt: "serotonin" } },
  { label: "PEN", params: { cell_type: "PEN" } },
  { label: "clear", params: null },
];

/** The sugar experiment of the Shiu et al. 2024 example: 20 sugar-sensing neurons at 100 Hz, MN9 as readout. */
const SUGAR_RUN = { preset: "sugar", readout_preset: "mn9", rate_hz: 100, duration_ms: 1000, trials: 3 };

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
let brainScene: BrainScene | null = null;
let pointCount: number | null = null; // known once the brain is loaded
let health: Health | null = null;
let streaming = false;
let selectedModel: string | null = null;
let player: Player | null = null;
let simulating = false;
let sugarChip: HTMLButtonElement | null = null;
let waveLegend = ""; // the legend line of the wave mode (restored when switching back to it)
let pathShown = false; // cables are on screen (the legend shows their summary)
/** Shared by chips and chat: whichever selection was requested last wins. */
const latest = new Latest();

/** Light up row indices (replacing any previous highlight). */
export function highlight(indices: ArrayLike<number>): void {
  brainScene?.setPath(null); // a new selection replaces the cables
  pathShown = false;
  brainScene?.setHighlight(indices);
}

export function clearAll(): void {
  waveLegend = "";
  brainScene?.setHighlight(null);
  brainScene?.setPath(null);
  pathShown = false;
  player?.clear();
}

/** Legend line while an activity wave is showing (lit dot, like a selection). */
function renderActivityLegend(text: string): void {
  $("legend-text").textContent = text;
  $("legend-selected").classList.add("on");
}

/** Show a recording; a silenced run also fetches its baseline so the difference map is available. */
async function showActivity(a: Activity, text: string): Promise<void> {
  waveLegend = text;
  let baseline: Activity | null = null;
  if (a.compareId) baseline = await loadActivity(a.compareId).catch(() => null); // expired baseline: just no diff mode
  player?.load(a, baseline);
  setActiveChip(null);
  renderActivityLegend(text);
}

/** "give sugar": POST /api/simulate (no LLM), then fetch the frames and play them. */
async function giveSugar(): Promise<void> {
  if (simulating || !player) return;
  simulating = true;
  if (sugarChip) sugarChip.disabled = true;
  latest.invalidate(); // a search finishing later must not replace the wave
  const started = performance.now();
  const pending = () => renderLegend(null, formatSimulating(SUGAR_RUN.duration_ms, (performance.now() - started) / 1000));
  pending();
  const tick = window.setInterval(pending, 1000);
  try {
    const res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(SUGAR_RUN),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`);
    const sim = parseSimulation(body);
    if (typeof sim === "string") throw new Error(`unexpected simulation answer: ${sim}`);
    const activity = await loadActivity(sim.simulationId);
    await showActivity(activity, activityLegend(sim.active, activity.nRows, sim.readoutHz === null ? null : { label: "MN9", hz: sim.readoutHz }));
    setActiveChip("give sugar");
  } catch (e) {
    renderLegend(null, `simulation failed: ${(e as Error).message}`);
  } finally {
    window.clearInterval(tick);
    simulating = false;
    if (sugarChip) sugarChip.disabled = health?.simulation === false;
  }
}

/** Resolve a highlight payload to row indices (inline, or fetched as binary). */
export async function resolveIndices(p: { indices: number[] | null; fetch?: string }): Promise<ArrayLike<number>> {
  return p.indices ?? (p.fetch ? await fetchIndices(p.fetch) : []);
}

function renderLegend(total: number | null, message?: string): void {
  const v = legendView(total, message);
  $("legend-text").textContent = v.text;
  $("legend-selected").classList.toggle("on", v.active);
}

function setActiveChip(label: string | null): void {
  for (const b of document.querySelectorAll<HTMLButtonElement>("#chips .chip")) {
    b.classList.toggle("active", b.textContent === label);
  }
}

function renderStatus(modelEl: HTMLElement): void {
  const v = deriveStatus(health, streaming, selectedModel);
  const dl = $("status");
  dl.replaceChildren(
    ...v.rows.map((r) => {
      const row = document.createElement("div");
      row.className = "status-row";
      const k = document.createElement("dt");
      k.textContent = r.key;
      const val = document.createElement("dd");
      val.textContent = r.value;
      val.dataset.state = r.value;
      row.append(k, val);
      return row;
    }),
  );
  modelEl.textContent = v.model;
}

function setupChips(): void {
  const sugar = document.createElement("button");
  sugar.className = "chip";
  sugar.textContent = "give sugar";
  sugar.title = "simulate 1 s of the fly tasting sugar (Shiu et al. 2024 model): 20 sugar-sensing neurons at 100 Hz, MN9 read out";
  sugar.addEventListener("click", () => void giveSugar());
  sugarChip = sugar;
  $("chips").appendChild(sugar);
  for (const chip of CHIPS) {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = chip.label;
    b.addEventListener("click", async () => {
      if (!chip.params) {
        latest.invalidate();
        clearAll();
        setActiveChip(null);
        renderLegend(null);
        return;
      }
      try {
        await runLatest(
          latest,
          async () => {
            const res = await fetch(`/api/neurons/search?${new URLSearchParams(chip.params!)}`);
            const body = await res.json();
            if (!res.ok) throw new Error(body.detail ?? `HTTP ${res.status}`);
            return { body, idx: await resolveIndices(body) };
          },
          ({ body, idx }) => {
            highlight(idx);
            setActiveChip(chip.label);
            renderLegend(body.total ?? 0, body.total ? undefined : (body.hint ?? undefined));
          },
        );
      } catch (e) {
        renderLegend(null, `search failed: ${(e as Error).message}`);
      }
    });
    $("chips").appendChild(b);
  }
}

/** Compact key for the current color mode: one row per category with its real neuron count. */
function renderKey(rows: { name: string; count: number; hex: string }[]): void {
  $("key").replaceChildren(
    ...rows.map((r) => {
      const row = document.createElement("div");
      row.className = "key-row";
      const dot = document.createElement("i");
      dot.style.background = r.hex;
      const name = document.createElement("span");
      name.textContent = r.name;
      const n = document.createElement("b");
      n.textContent = r.count.toLocaleString("en-US");
      row.append(dot, name, n);
      return row;
    }),
  );
}

/** Mode, orbit and camera chips, shown once the scene is up. */
function setupViewChips(scene: BrainScene, theme: Theme, regions?: Regions, nt?: Transmitters): void {
  const mk = (label: string, title: string, onClick: (b: HTMLButtonElement) => void, extra = "") => {
    const b = document.createElement("button");
    b.className = `chip ${extra}`.trim();
    b.textContent = label;
    b.title = title;
    b.addEventListener("click", () => onClick(b));
    $("views").appendChild(b);
    return b;
  };
  const regionRows = () =>
    regions ? keyRows(regions.codes, regions.table, (n) => theme.tones[n] ?? theme.baseHex) : [];
  const ntRows = () =>
    nt ? keyRows(nt.codes, nt.table, (n) => greyHex(ntTier(n).tone)).sort((a, b) => ntTier(b.name).tone - ntTier(a.name).tone) : [];
  if (scene.hasTransmitters) {
    const modes: HTMLButtonElement[] = [];
    const select = (mode: "regions" | "transmitters") => {
      scene.setColorMode(mode);
      renderKey(mode === "regions" ? regionRows() : ntRows());
      modes.forEach((m) => m.classList.toggle("active", m.textContent === mode));
    };
    modes.push(mk("regions", "tint the neurons by brain region", () => select("regions"), "mode active"));
    modes.push(mk("transmitters", "tint the neurons by predicted neurotransmitter", () => select("transmitters"), "mode"));
  }
  renderKey(regionRows());
  mk("orbit", "slow auto-orbit; pauses while you drag", (b) => {
    scene.setOrbit(!scene.orbitOn);
    b.classList.toggle("active", scene.orbitOn);
  }, `sep toggle${scene.orbitOn ? " active" : ""}`);
  mk("front", "reset the camera to a straight front view", () => scene.setView("front"), "sep");
  mk("3/4", "reset the camera to the default three-quarter view", () => scene.setView("home"));
}

async function loadHealth(): Promise<void> {
  try {
    const res = await fetch("/api/health");
    health = res.ok ? parseHealth(await res.json()) : null;
  } catch {
    health = null;
  }
}

async function loadModels(): Promise<ReturnType<typeof parseModels>> {
  try {
    const res = await fetch("/api/models");
    return res.ok ? parseModels(await res.json()) : null;
  } catch {
    return null; // older backend: no picker, the server default is used
  }
}

async function boot(): Promise<void> {
  const theme = readTheme();
  $("version").textContent = versionLine(null);
  renderLegend(null);
  setupChips();
  const chat = mountChat($("chat"), {
    onHighlight: (p) =>
      runLatest(latest, () => resolveIndices(p), (idx) => {
        highlight(idx);
        setActiveChip(null);
        renderLegend(p.total);
      }).then(() => undefined),
    onPath: (p) => {
      brainScene?.setPath(p);
      pathShown = true;
      renderActivityLegend(pathLegend(p)); // lit dot + the route's summary
    },
    onNewChat: () => {
      if (!pathShown) return;
      brainScene?.setPath(null);
      pathShown = false;
      renderLegend(null);
    },
    onActivity: async (a) => {
      const activity = await loadActivity(a.simulationId);
      await showActivity(activity, activityLegend(activity.activeCount, activity.nRows, null));
    },
    getPointCount: () => pointCount,
    onModelChange: (id) => {
      selectedModel = id;
      renderStatus(chat.model);
    },
    onStreaming: (s) => {
      streaming = s;
      renderStatus(chat.model);
    },
  });
  renderStatus(chat.model);
  void loadModels().then((list) => chat.setModels(list));
  void loadHealth().then(() => {
    renderStatus(chat.model);
    if (sugarChip && health?.simulation === false) {
      sugarChip.disabled = true;
      sugarChip.title = "the simulator is not set up on this server (run flybrain-build-sim-weights)";
    }
  });
  try {
    const brain = await loadBrain();
    pointCount = brain.meta.count;
    // Region tones and landmarks are optional context: the cloud renders without them.
    const table = brain.meta.regions;
    const ntTable = brain.meta.nt;
    const [regions, landmarks, nt] = await Promise.allSettled([
      table ? loadRegions(brain.meta.count, table) : Promise.reject(new Error("no region table in /brain/meta")),
      loadLandmarks(),
      ntTable ? loadNt(brain.meta.count, ntTable) : Promise.reject(new Error("no transmitter table in /brain/meta")),
    ]);
    const ok = <T,>(r: PromiseSettledResult<T>) => (r.status === "fulfilled" ? r.value : undefined);
    brainScene = createScene($("stage"), brain, theme, { regions: ok(regions), landmarks: ok(landmarks), nt: ok(nt) });
    player = createPlayer({
      scene: brainScene,
      root: $("timeline"),
      reducedMotion: window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true,
      speed: theme.playbackSpeed,
      onMode: (_mode, legend) => {
        if (legend !== null) renderActivityLegend(legend);
        else if (waveLegend) renderActivityLegend(waveLegend);
      },
    });
    mountInspector({
      stage: $("stage"),
      scene: brainScene,
      pointCount,
      onShowPartners: (rows, label) => {
        latest.invalidate();
        highlight(rows);
        setActiveChip(null);
        renderLegend(null, label);
        $("legend-selected").classList.add("on");
      },
    });
    setupViewChips(brainScene, theme, ok(regions), ok(nt));
    $("version").textContent = versionLine(brain.meta.count);
  } catch (e) {
    $("version").textContent = `could not load the brain: ${(e as Error).message}`;
  }
}

void boot();
