import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "./style.css";
import { mountChat } from "./chat";
import { mountChatToggle } from "./chatToggle";
import { parseModels } from "./models";
import { fetchIndices, loadBrain, loadBrainMeta } from "./brainData";
import { mountPreview } from "./brainPreview";
import { createBrainGate } from "./brainGate";
import { mountOpenControl } from "./previewOpen";
import { mountBody, type MountedBody } from "./bodyMount";
import { createBodyAnimation, DEFAULT_BODY_CAPTION, type BodyAnimation } from "./bodyAnimation";
import { createBodyDriver } from "./bodyDriver";
import { loadWalkClip } from "./bodyClip";
import { createWalkPlayer, type WalkPlayer } from "./bodyWalk";
import { createClipPlayer, type ClipPlayer } from "./bodyClipPlayer";
import { createLiveClient, DEV_PRESET, DEV_RATE_HZ, type LiveClient } from "./liveClient";
import { createBodyFlow, type BodyFlow, type FlowKey } from "./bodyFlow";
import { mountLayout, type Layout } from "./layoutPanes";
import type { Pane } from "./layoutState";
import { commitScene } from "./sceneLoad";
import { shouldOpenOnBoot, shouldOpenOnChange } from "./sceneOpen";
import { PathOrder } from "./pathOrder";
import { chartsChipState, simChipsDisabled } from "./chipState";
import { afterNewChat, EMPTY, legendOf, modeStage, StageHolder, type Stage } from "./stage";
import { createScene, type BrainScene } from "./scene";
import { Latest, runLatest } from "./latest";
import { loadLatestActivity } from "./activityLoader";
import { deriveStatus, parseHealth, versionLine, type Health } from "./status";
import { readTheme } from "./theme";
import { loadLandmarks, loadRegions, type Regions } from "./regions";
import { activityLegend, loadActivity, type Activity } from "./activity";
import { createPlayer, type Player } from "./player";
import { createChartsPanel, type ChartsPanel } from "./chartsPanel";
import { mountInspector } from "./inspector";
import { pathLegend } from "./path";
import { type ExperimentChart } from "./experiments";
import { addGroup, addSubgroup, renderKey } from "./controls";
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

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
let brainScene: BrainScene | null = null;
let pointCount: number | null = null; // known once the brain is loaded
let health: Health | null = null;
let streaming = false;
let selectedModel: string | null = null;
let player: Player | null = null;
let mountedBody: MountedBody | null = null;
/** Recording -> body pose and caption; one per page, fed by showActivity/clearAll and by the player clock. */
let bodyAnimation: BodyAnimation | null = null;
/** The physics walking clip replayed on the body (a published controller, not driven by the brain); one per page. */
let walkPlayer: WalkPlayer | null = null;
let walkChip: HTMLButtonElement | null = null;
/** A body clip computed from the brain recording (our mapping of descending neurons), driven by the player's clock. */
let clipPlayer: ClipPlayer | null = null;
/** simulate (or take a recording) -> physics body -> play both; one per page. */
let bodyFlow: BodyFlow | null = null;
const flowChips = new Map<FlowKey, HTMLButtonElement>();
let moveChip: HTMLButtonElement | null = null;
/** DEV ENTRY of the live brain+body session (/ws/live): one discreet chip and `window.flyLive`, until the playground's
 *  agent tools (P4) replace them with chat-driven control. */
let liveClient: LiveClient | null = null;
let liveChip: HTMLButtonElement | null = null;
let activeFlowKey: FlowKey | null = null;
let charts: ChartsPanel | null = null;
let chartsChip: HTMLButtonElement | null = null;
let simulating = false;
let sugarChip: HTMLButtonElement | null = null;
let experimentChips: HTMLButtonElement[] = [];
let waveLegend = ""; // the legend line of the wave mode (restored when switching back to it)
let pathShown = false; // cables are on screen (the scene layer; independent of what the legend says)
/** Ordering of the cables: the newest path wins, and anything that replaces them cancels a path still waiting for the scene. */
const pathOrder = new PathOrder();
/** What the legend line says (one owner; the DOM is only a rendering of it). */
const stageHolder = new StageHolder();
let chartsHasContent = false; // a recording or an experiment chart is available to plot
let lastMain: Pane = "body"; // the pane that was large before the last layout change
/** The recording currently on stage; replayed into the player when the scene appears later (never lost). */
let shown: { a: Activity; baseline: Activity | null } | null = null;
let theme: Theme;
/** The real scene is built only on demand (all-or-nothing, never unloaded). Everything that needs it goes through here. */
const gate = createBrainGate<BrainScene>({ load: loadScene });
/** Shared by chips and chat: whichever selection was requested last wins. */
const latest = new Latest();
/** Recordings have their own token: the newest requested one is the one shown, whatever order the loads finish in. */
const activityLatest = new Latest();

/** Light up row indices (replacing any previous highlight). */
export function highlight(indices: ArrayLike<number>): void {
  brainScene?.setPath(null); // a new selection replaces the cables
  pathShown = false;
  pathOrder.cancel(); // a path still waiting for the scene must not draw over this selection
  brainScene?.setHighlight(indices);
}

export function clearAll(): void {
  activityLatest.invalidate();
  waveLegend = "";
  charts?.clear();
  chartsHasContent = false;
  syncChartsChip();
  brainScene?.setHighlight(null);
  brainScene?.setPath(null);
  pathShown = false;
  pathOrder.cancel(); // a path still waiting for the scene is dropped
  shown = null;
  walkPlayer?.stop(); // a cleared stage is not the walking demo either
  bodyFlow?.stop();
  bodyAnimation?.setActivity(null); // before player.clear(): its null tick then finds nothing loaded (rest, default caption)
  player?.clear();
}

/** Take the legend: record the new stage, then render it. */
function setStage(s: Stage): void {
  stageHolder.set(s);
  paintLegend(s);
}

function paintLegend(s: Stage): void {
  const v = legendOf(s);
  $("legend-text").textContent = v.text;
  $("legend-selected").classList.toggle("on", v.active);
}

/** The charts chip is derived: enabled when there is something to plot, active when the panel is open. */
function syncChartsChip(): void {
  if (!chartsChip) return;
  const v = chartsChipState({ hasContent: chartsHasContent, panelOpen: charts?.open === true });
  chartsChip.disabled = v.disabled;
  chartsChip.classList.toggle("active", v.active);
}

/** The simulator chips are derived from the run in flight, the chat stream and the server's capability. */
function renderSimChips(): void {
  const flowBusy = bodyFlow?.state === "simulating" || bodyFlow?.state === "computing";
  const off = simChipsDisabled({ simulating: simulating || flowBusy, streaming, simulationOff: health?.simulation === false });
  for (const b of simChips()) b.disabled = off;
  syncFlowChips();
}

/** The brain -> body chips: the one that runs is a stop button; the others wait for it. "move body" needs a recording. */
function syncFlowChips(): void {
  const state = bodyFlow?.state ?? "idle";
  const busy = state === "simulating" || state === "computing";
  const off = simChipsDisabled({ simulating, streaming, simulationOff: health?.simulation === false });
  for (const [key, b] of flowChips) {
    const mine = activeFlowKey === key && state !== "idle";
    b.classList.toggle("active", mine);
    b.setAttribute("aria-pressed", String(mine));
    b.disabled = mine ? false : off || busy;
  }
  if (moveChip) moveChip.disabled = off || busy || !shown?.a.id;
}

/** A recording plus, for a silenced run, its baseline (so the difference map is available). */
async function loadWithBaseline(id: string): Promise<{ a: Activity; baseline: Activity | null }> {
  const a = await loadActivity(id);
  const baseline = a.compareId ? await loadActivity(a.compareId).catch(() => null) : null; // expired baseline: just no diff mode
  return { a, baseline };
}

function showActivity({ a, baseline }: { a: Activity; baseline: Activity | null }, text: string): void {
  waveLegend = text;
  pathOrder.cancel(); // a recording replaces the cables' story: a path still waiting must not draw
  shown = { a, baseline };
  walkPlayer?.stop(); // the recording owns the body and its caption now
  bodyFlow?.supersede(); // a body still being computed for an older recording is dropped...
  clipPlayer?.clear(); // ...and so is the clip on the body (the flow sets its own again after showing its recording)
  bodyAnimation?.setActivity(a); // the MAIN recording, never the baseline; before player.load so its first tick is already driven
  player?.load(a, baseline); // no player yet: loadScene replays `shown` once the scene exists
  charts?.show(a, baseline);
  chartsHasContent = true;
  syncChartsChip();
  setActiveChip(null);
  setStage({ kind: "wave", text });
  syncFlowChips();
}

/** Chips that run the simulator (one at a time). */
function simChips(): HTMLButtonElement[] {
  return [...(sugarChip ? [sugarChip] : []), ...experimentChips];
}

/** Plot an experiment chart in the charts panel (opening it) and offer its rows' difference maps. */
function showExperimentChart(c: ExperimentChart): void {
  charts?.showExperiment(c, (simulationId) => {
    void loadLatestActivity(activityLatest, () => loadWithBaseline(simulationId), (loaded) =>
      showActivity(loaded, activityLegend(loaded.a.activeCount, loaded.a.nRows, null))).catch((e) =>
      renderLegend(null, `could not open that run: ${(e as Error).message}`));
  });
  charts?.setOpen(true);
  chartsHasContent = true;
  syncChartsChip();
}

/** Resolve a highlight payload to row indices (inline, or fetched as binary). */
export async function resolveIndices(p: { indices: number[] | null; fetch?: string }): Promise<ArrayLike<number>> {
  return p.indices ?? (p.fetch ? await fetchIndices(p.fetch) : []);
}

function renderLegend(total: number | null, message?: string): void {
  setStage(total === null ? (message ? { kind: "message", text: message } : EMPTY) : { kind: "selection", total, msg: message });
}

function setActiveChip(label: string | null): void {
  for (const b of document.querySelectorAll<HTMLButtonElement>("#chips .chip")) {
    if (b === walkChip || [...flowChips.values()].includes(b)) continue; // their pressed state follows the player / the flow, not the recordings
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
  const runItems = addGroup($("chips"), "run", "RUN");
  const hlItems = addGroup($("chips"), "highlight", "HIGHLIGHT");
  const liveBtn = document.createElement("button");
  liveBtn.className = "chip run";
  liveBtn.textContent = "live";
  liveBtn.title = `dev entry: start or stop a live brain+body session in the glass cube (${DEV_PRESET} at ${DEV_RATE_HZ} Hz). Shows its real speed; the chat adds cubes and threats to it`;
  liveBtn.setAttribute("aria-pressed", "false");
  liveBtn.addEventListener("click", () => liveClient?.toggle());
  liveChip = liveBtn;
  runItems.appendChild(liveBtn);
  for (const chip of CHIPS) {
    const b = document.createElement("button");
    b.className = chip.params ? "chip" : "chip sep";
    b.textContent = chip.label;
    b.addEventListener("click", async () => {
      if (chip.params) openBrain();
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
            const idx = await resolveIndices(body);
            await gate.ensure();
            return { body, idx };
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
    hlItems.appendChild(b);
  }
  renderSimChips();
}

/** Mode, orbit and camera chips, shown once the scene is up. */
function setupViewChips(scene: BrainScene, theme: Theme, regions?: Regions, nt?: Transmitters): () => void {
  const viewItems = addGroup($("views"), "view", "VIEW");
  const mk = (label: string, title: string, onClick: (b: HTMLButtonElement) => void, extra = "", into: HTMLElement = viewItems) => {
    const b = document.createElement("button");
    b.className = `chip ${extra}`.trim();
    b.textContent = label;
    b.title = title;
    b.addEventListener("click", () => onClick(b));
    into.appendChild(b);
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
      renderKey($("key"), mode === "regions" ? "REGIONS" : "TRANSMITTERS", mode === "regions" ? regionRows() : ntRows());
      modes.forEach((m) => m.classList.toggle("active", m.textContent === mode));
    };
    modes.push(mk("regions", "tint the neurons by brain region", () => select("regions"), "mode active"));
    modes.push(mk("transmitters", "tint the neurons by predicted neurotransmitter", () => select("transmitters"), "mode"));
  }
  renderKey($("key"), "REGIONS", regionRows());
  chartsChip = mk("charts", "spike raster of the busiest neurons and the readout trace over time", () => {
    charts?.setOpen(!charts.open);
    syncChartsChip();
  }, "toggle");
  syncChartsChip(); // needs a recording; active if the panel was opened (by an experiment) before the scene existed
  mk("orbit", "slow auto-orbit; pauses while you drag", (b) => {
    scene.setOrbit(!scene.orbitOn);
    b.classList.toggle("active", scene.orbitOn);
  }, `toggle${scene.orbitOn ? " active" : ""}`);
  const camera = addSubgroup(viewItems, "camera");
  mk("front", "reset the camera to a straight front view", () => scene.setView("front"), "", camera);
  mk("3/4", "reset the camera to the default three-quarter view", () => scene.setView("home"), "", camera);
  return () => {
    viewItems.parentElement?.remove(); // the whole VIEW group
    chartsChip = null;
  };
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

/** Fire-and-forget open for actions whose result is replayed or tolerated without the scene (failure shows in the preview hint). */
function openBrain(): void {
  void gate.ensure().catch(() => undefined);
}

/** Everything that happens when the real scene is wanted: the brain, the optional layers, the scene and its wiring. */
async function loadScene(): Promise<BrainScene> {
  try {
    const brain = await loadBrain();
    // Region tones and landmarks are optional context: the cloud renders without them.
    const table = brain.meta.regions;
    const ntTable = brain.meta.nt;
    const [regions, landmarks, nt] = await Promise.allSettled([
      table ? loadRegions(brain.meta.count, table) : Promise.reject(new Error("no region table in /brain/meta")),
      loadLandmarks(),
      ntTable ? loadNt(brain.meta.count, ntTable) : Promise.reject(new Error("no transmitter table in /brain/meta")),
    ]);
    const ok = <T,>(r: PromiseSettledResult<T>) => (r.status === "fulfilled" ? r.value : undefined);
    // All-or-nothing: wire everything first, publish (brainScene, placeholders removed) last. A failure rolls the
    // wiring back and disposes the scene, so the preview and its retry control stay and a retry builds ONE scene.
    let newPlayer: Player | null = null;
    return commitScene<BrainScene>({
      create: () => createScene(layout.brainHost, brain, theme, { regions: ok(regions), landmarks: ok(landmarks), nt: ok(nt) }),
      wire: (scene, undo) => {
        const inspector = mountInspector({
          stage: layout.brainHost, // the card is placed in the same box as the canvas (the brain pane), not the whole stage
          scene,
          pointCount: brain.meta.count,
          onShowPartners: (rows, label) => {
            latest.invalidate();
            highlight(rows);
            setActiveChip(null);
            setStage({ kind: "selection", total: null, msg: label, lit: true });
          },
        });
        undo(() => inspector.dispose());
        undo(setupViewChips(scene, theme, ok(regions), ok(nt)));
        newPlayer = createPlayer({
          scene,
          root: $("timeline"),
          reducedMotion: window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true,
          speed: theme.playbackSpeed,
          onTime: (ms) => charts?.setTime(ms),
          onMode: (_mode, legend) => {
            const next = modeStage(stageHolder.current, legend, waveLegend);
            if (next !== stageHolder.current) setStage(next);
          },
        });
        const p = newPlayer;
        undo(() => p.clear());
        // The body follows the playback clock (it runs even while the brain pane is hidden). Subscribed once per player, before the replay below.
        undo(p.subscribeTime((ms) => { bodyAnimation?.onTime(ms); clipPlayer?.onTime(ms); }));
        if (shown) p.load(shown.a, shown.baseline); // a recording that arrived before the scene existed
      },
      publish: (scene) => {
        brainScene = scene;
        pointCount = brain.meta.count;
        player = newPlayer;
        openControl?.remove();
        preview?.remove();
        $("version").textContent = versionLine(brain.meta.count);
      },
    });
  } catch (e) {
    $("version").textContent = `could not load the brain: ${(e as Error).message}`;
    throw e;
  }
}

/** The stage panes (body, brain); created first thing in boot(), before anything can ask for the real scene. */
let layout: Layout;
let preview: ReturnType<typeof mountPreview> | null = null;
let openControl: ReturnType<typeof mountOpenControl> | null = null;

async function boot(): Promise<void> {
  theme = readTheme();
  // Body large and brain inset by default, remembered. Swapping is a CSS class change: no scene is rebuilt.
  layout = mountLayout($("stage"), {
    // showing the brain large from its preview means the user wants the real thing: load it (idempotent)
    onChange: (s) => {
      if (shouldOpenOnChange(lastMain, s)) openBrain();
      lastMain = s.main;
    },
  });
  lastMain = layout.state().main;
  $("version").textContent = versionLine(null);
  // the neuron count comes from the small meta document: it must not wait for the (on-demand) positions
  void loadBrainMeta()
    .then((meta) => {
      if (!brainScene) $("version").textContent = versionLine(meta.count);
    })
    .catch(() => {}); // the preview and the real load report their own errors
  renderLegend(null);
  setupChips();
  charts = createChartsPanel($("charts")); // charts do not need the scene
  const chat = mountChat($("chat"), {
    // Events that need the scene wait for it (gate.ensure) instead of being dropped. While it loads only the
    // NEWEST of each kind survives: highlight and activity through their Latest tokens, the path through pathOrder.
    onHighlight: (p) =>
      runLatest(latest, async () => {
        const idx = await resolveIndices(p);
        await gate.ensure();
        // the point count was unknown when the event was parsed: check the range now
        if (pointCount !== null) for (let i = 0; i < idx.length; i++) if (idx[i] >= pointCount) throw new Error(`index ${idx[i]} is outside 0..${pointCount - 1}`);
        return idx;
      }, (idx) => {
        highlight(idx);
        setActiveChip(null);
        renderLegend(p.total);
      }).then(
        (applied) => pathOrder.highlightSettled(applied), // false: a newer selection took over, its cables must not be drawn
        (e) => {
          pathOrder.highlightSettled(true); // a failed (not superseded) highlight still lets its path draw
          throw e;
        },
      ),
    onPath: async (p) => {
      const token = pathOrder.begin();
      if (token === null) return; // its own highlight was superseded
      const scene = await gate.ensure();
      if (!pathOrder.isCurrent(token)) return; // a newer path, a highlight, a recording, a new chat or a clear came meanwhile
      if (pointCount !== null && p.rows.some((r) => r >= pointCount!)) throw new Error("path row outside the brain");
      scene.setPath(p);
      pathShown = true;
      setStage({ kind: "path", text: pathLegend(p) }); // lit dot + the route's summary
    },
    onNewChat: () => {
      pathOrder.cancel();
      if (pathShown) {
        brainScene?.setPath(null);
        pathShown = false;
      }
      const next = afterNewChat(stageHolder.current); // only a path summary is cleared, never a recording's legend
      if (next !== stageHolder.current) setStage(next);
    },
    onActivity: async (a) => {
      // The recording is shown even if the scene cannot load (charts work); `shown` replays it into a later scene.
      openBrain();
      await loadLatestActivity(activityLatest, () => loadWithBaseline(a.simulationId), (loaded) =>
        showActivity(loaded, activityLegend(loaded.a.activeCount, loaded.a.nRows, null)));
    },
    onChart: showExperimentChart,
    getPointCount: () => pointCount,
    onModelChange: (id) => {
      selectedModel = id;
      renderStatus(chat.model);
    },
    onStreaming: (s) => {
      streaming = s;
      renderStatus(chat.model);
      renderSimChips();
    },
  });
  mountChatToggle(document.body, $("chat")); // hide/show the panel; the conversation DOM stays mounted
  renderStatus(chat.model);
  void loadModels().then((list) => chat.setModels(list));
  void loadHealth().then(() => {
    renderStatus(chat.model);
    if (health?.simulation === false) {
      for (const b of simChips()) b.title = "the simulator is not set up on this server (run flybrain-build-sim-weights)";
    }
    renderSimChips();
  });
  // The fly body is the default center pane; the brain preview (then the real scene, on demand) lives in the brain pane.
  const calm = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;
  mountedBody = mountBody(layout.bodyHost, { theme, reducedMotion: calm });
  const body = mountedBody;
  bodyAnimation = createBodyAnimation({
    driver: createBodyDriver({ setPose: (angles) => body.setPose(angles) }),
    setCaption: (text) => body.setCaption(text),
    defaultCaption: DEFAULT_BODY_CAPTION,
    reducedMotion: calm,
  });
  walkPlayer = createWalkPlayer({
    load: () => loadWalkClip(),
    setFrame: (mats) => body.setClipFrame(mats),
    setCaption: (text) => body.setCaption(text),
    defaultCaption: DEFAULT_BODY_CAPTION,
    schedule: (cb) => requestAnimationFrame(cb),
    cancel: (id) => cancelAnimationFrame(id),
    reducedMotion: calm,
    onChange: (on) => {
      walkChip?.classList.toggle("active", on);
      walkChip?.setAttribute("aria-pressed", String(on));
    },
  });
  clipPlayer = createClipPlayer({
    setFrame: (mats) => body.setClipFrame(mats),
    setWorld: (w) => body.setWorld(w),
    setPose: (angles) => body.setPose(angles),
    reducedMotion: calm,
  });
  bodyFlow = createBodyFlow({
    request: (url, init) => fetch(url, init),
    showRecording: async (id) => {
      const legend = (l: { a: Activity }) => activityLegend(l.a.activeCount, l.a.nRows, null);
      return loadLatestActivity(activityLatest, () => loadWithBaseline(id), (loaded) => showActivity(loaded, legend(loaded)));
    },
    play: (clip, caption) => {
      bodyAnimation?.setActivity(null); // the clip owns the pose and the caption (the recording's own readout pose would fight it)
      body.setCaption(caption);
      clipPlayer?.setClip(clip);
    },
    clearBody: () => {
      clipPlayer?.clear();
      bodyAnimation?.setActivity(shown?.a ?? null); // back to the recording's own readout pose and caption, or rest
    },
    setStatus: (text) => body.setCaption(text ?? DEFAULT_BODY_CAPTION),
    onState: () => renderSimChips(),
  });
  liveClient = createLiveClient({
    body: {
      setFrame: (mats) => body.setClipFrame(mats), setWorld: (w) => body.setWorld(w), setPose: (a) => body.setPose(a),
      setCaption: (t) => body.setCaption(t), setSandbox: (on) => body.setSandbox(on),
      setArena: (a) => body.setArena(a),
      setObjects: (o) => { body.setObjects(o); },
    },
    defaultCaption: DEFAULT_BODY_CAPTION,
    prepare: () => { // the live session owns the body now
      bodyFlow?.stop();
      walkPlayer?.stop();
      clipPlayer?.clear();
      bodyAnimation?.setActivity(null);
    },
  });
  liveClient.onChange((st) => {
    const on = st.state !== "idle" && st.state !== "closed";
    liveChip?.classList.toggle("active", on);
    liveChip?.setAttribute("aria-pressed", String(on));
    if (st.state === "idle") bodyAnimation?.setActivity(shown?.a ?? null); // back to the recording's own readout pose and caption, or rest
  });
  (window as unknown as { flyLive?: unknown }).flyLive = {
    start: (preset: string | null = DEV_PRESET, rateHz: number = DEV_RATE_HZ) => liveClient?.start(preset, rateHz),
    stop: () => liveClient?.stop(),
    stimulus: (preset: string | null, rateHz: number) => liveClient?.stimulus(preset, rateHz),
    status: () => liveClient?.status,
  };
  syncFlowChips();
  preview = mountPreview(layout.brainHost, { theme });
  openControl = mountOpenControl(layout.brainHost, gate);
  // the layout only reports changes, never its initial state: a remembered "brain large" opens the scene here
  if (shouldOpenOnBoot(layout.state())) openBrain();
}

void boot();
