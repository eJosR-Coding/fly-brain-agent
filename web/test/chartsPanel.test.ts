// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Activity } from "../src/activity";
import { createChartsPanel } from "../src/chartsPanel";

function activity(): Activity {
  const counts = new Uint8Array(20);
  counts[1] = 2;
  return {
    rows: new Uint32Array([0, 1]), counts, nRows: 2, nBins: 10, binMs: 10, durationMs: 100, peak: 2, activeCount: 1,
    id: null, trials: 1, compareId: null, silencedRows: null,
    readout: { label: "MN9", neurons: 1, rows: [0], hz: Array(10).fill(5) },
  };
}

const heads = (root: HTMLElement) => [...root.querySelectorAll<SVGLineElement>("line.ch-head")];
const flush = () => new Promise((r) => setTimeout(r, 0));

function setup() {
  const root = document.createElement("div");
  document.body.appendChild(root);
  const panel = createChartsPanel(root);
  return { root, panel };
}

afterEach(() => {
  vi.unstubAllGlobals();
  document.body.replaceChildren();
});

describe("charts playhead keeps the last player time", () => {
  it("shows a time set before the charts were drawn", async () => {
    const { root, panel } = setup();
    panel.setTime(50);
    panel.show(activity(), null);
    panel.setOpen(true);
    await flush();
    const hs = heads(root);
    expect(hs).toHaveLength(2);
    for (const h of hs) {
      expect(h.getAttribute("visibility")).toBe("visible");
      expect(Number(h.getAttribute("x1"))).toBeGreaterThan(84);
    }
  });

  it("re-applies the time after a redraw (close and reopen while the player is idle)", async () => {
    const { root, panel } = setup();
    panel.show(activity(), null);
    panel.setOpen(true);
    await flush();
    panel.setTime(100);
    const x = heads(root)[0].getAttribute("x1");
    panel.setOpen(false);
    panel.setOpen(true);
    await flush();
    const h = heads(root)[0];
    expect(h.getAttribute("visibility")).toBe("visible");
    expect(h.getAttribute("x1")).toBe(x);
  });

  it("keeps the playhead hidden when the last time was null", async () => {
    const { root, panel } = setup();
    panel.show(activity(), null);
    panel.setOpen(true);
    await flush();
    panel.setTime(40);
    panel.setTime(null);
    panel.setOpen(false);
    panel.setOpen(true);
    await flush();
    for (const h of heads(root)) expect(h.getAttribute("visibility")).toBe("hidden");
  });

  it("clear forgets the time: the next recording starts with a hidden playhead", async () => {
    const { root, panel } = setup();
    panel.setTime(60);
    panel.show(activity(), null);
    panel.setOpen(true);
    await flush();
    panel.clear();
    expect(() => panel.setTime(null)).not.toThrow();
    panel.show(activity(), null);
    await flush();
    const hs = heads(root);
    expect(hs).toHaveLength(2);
    for (const h of hs) expect(h.getAttribute("visibility")).toBe("hidden");
  });
});
