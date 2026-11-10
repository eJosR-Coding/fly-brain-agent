import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "./style.css";
import { mountChat } from "./chat";
import { fetchIndices, loadBrain } from "./brainData";
import { createScene, type BrainScene } from "./scene";
import { Latest, runLatest } from "./latest";
import { deriveStatus, legendView, parseHealth, versionLine, type Health } from "./status";
import { readTheme } from "./theme";
import { loadLandmarks, loadRegions, type Regions } from "./regions";
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
/** Shared by chips and chat: whichever selection was requested last wins. */
const latest = new Latest();

/** Light up row indices (replacing any previous highlight). */
export function highlight(indices: ArrayLike<number>): void {
  brainScene?.setHighlight(indices);
}

export function clearAll(): void {
  brainScene?.setHighlight(null);
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
  const v = deriveStatus(health, streaming);
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
    getPointCount: () => pointCount,
    onStreaming: (s) => {
      streaming = s;
      renderStatus(chat.model);
    },
  });
  renderStatus(chat.model);
  void loadHealth().then(() => renderStatus(chat.model));
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
    setupViewChips(brainScene, theme, ok(regions), ok(nt));
    $("version").textContent = versionLine(brain.meta.count);
  } catch (e) {
    $("version").textContent = `could not load the brain: ${(e as Error).message}`;
  }
}

void boot();
