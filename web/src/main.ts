import "./style.css";
import { mountChat } from "./chat";
import { fetchIndices, loadBrain } from "./brainData";
import { applyHighlight, clearHighlight } from "./colors";
import { createScene, type BrainScene } from "./scene";
import { Latest, runLatest } from "./latest";
import { readTheme, type Theme } from "./theme";

const CHIPS: { label: string; params: Record<string, string> | null }[] = [
  { label: "dopamine", params: { nt: "dopamine" } },
  { label: "dopamine left", params: { nt: "dopamine", side: "left" } },
  { label: "GABA", params: { nt: "gaba" } },
  { label: "serotonin", params: { nt: "serotonin" } },
  { label: "PEN", params: { cell_type: "PEN" } },
  { label: "clear", params: null },
];

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
let brainScene: BrainScene | null = null;
let theme: Theme;
/** Shared by chips and chat: whichever selection was requested last wins. */
const latest = new Latest();

/** Light up row indices (replacing any previous highlight). */
export function highlight(indices: ArrayLike<number>): void {
  if (!brainScene) return;
  clearHighlight(brainScene.colors, brainScene.base);
  applyHighlight(brainScene.colors, indices, theme.highlightRgb);
  brainScene.refreshColors();
}

export function clearAll(): void {
  if (!brainScene) return;
  clearHighlight(brainScene.colors, brainScene.base);
  brainScene.refreshColors();
}

/** Resolve a highlight payload to row indices (inline, or fetched as binary). */
export async function resolveIndices(p: { indices: number[] | null; fetch?: string }): Promise<ArrayLike<number>> {
  return p.indices ?? (p.fetch ? await fetchIndices(p.fetch) : []);
}

function setupChips(): void {
  const sel = $("selection");
  for (const chip of CHIPS) {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = chip.label;
    b.addEventListener("click", async () => {
      if (!chip.params) {
        latest.invalidate();
        clearAll();
        sel.textContent = "";
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
            sel.textContent = body.total ? `${body.total.toLocaleString("en-US")} neurons: ${chip.label}` : (body.hint ?? "no matches");
          },
        );
      } catch (e) {
        sel.textContent = `search failed: ${(e as Error).message}`;
      }
    });
    $("chips").appendChild(b);
  }
}

async function boot(): Promise<void> {
  theme = readTheme();
  setupChips();
  mountChat($("chat"), (p) =>
    runLatest(latest, () => resolveIndices(p), (idx) => {
      highlight(idx);
      $("selection").textContent = `${p.total.toLocaleString("en-US")} neurons: from chat`;
    }).then(() => undefined),
  );
  try {
    const brain = await loadBrain();
    brainScene = createScene($("stage"), brain, theme);
    $("status").textContent = `${brain.meta.count.toLocaleString("en-US")} neurons loaded`;
  } catch (e) {
    $("status").textContent = `could not load the brain: ${(e as Error).message}`;
  }
}

void boot();
