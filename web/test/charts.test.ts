import { describe, expect, it } from "vitest";
import { decodeFrames, type Activity } from "../src/activity";
import {
  groupLabelLayout, linearScale, niceTicks, rasterModel, readoutSeries, rasterTooltip, traceModel, tracePath,
  traceTooltip, parseRowTypes, smooth, smoothWindow,
} from "../src/charts";

function wire(rows: number[], counts: number[][]): ArrayBuffer {
  const nBins = counts[0].length;
  const buf = new ArrayBuffer(rows.length * 4 + rows.length * nBins);
  const view = new DataView(buf);
  rows.forEach((r, i) => view.setUint32(i * 4, r, true));
  new Uint8Array(buf, rows.length * 4).set(counts.flat());
  return buf;
}
const act = (rows: number[], counts: number[][], trials = 1, binMs = 100): Activity => ({
  ...decodeFrames(wire(rows, counts), rows.length, counts[0].length, binMs),
  trials,
});

describe("linearScale", () => {
  it("maps the domain onto the range and clamps", () => {
    const s = linearScale(0, 1000, 10, 210);
    expect(s(0)).toBe(10);
    expect(s(500)).toBe(110);
    expect(s(2000)).toBe(210);
    expect(s(-5)).toBe(10);
  });
  it("survives an empty domain", () => {
    expect(linearScale(5, 5, 0, 100)(5)).toBe(0);
  });
});

describe("niceTicks", () => {
  it("rounds the top up to a 1-2-5 step and lists the ticks from zero", () => {
    expect(niceTicks(61, 4)).toEqual({ max: 80, ticks: [0, 20, 40, 60, 80] });
    expect(niceTicks(9, 3)).toEqual({ max: 10, ticks: [0, 5, 10] });
  });
  it("never returns a zero-height axis", () => {
    expect(niceTicks(0, 3).max).toBeGreaterThan(0);
  });
});

describe("rasterModel", () => {
  // rows 10..14; row 12 is the most active, 14 the least. 10 and 11 share a cell type.
  const a = act([10, 11, 12, 13, 14], [[1, 0, 0], [1, 1, 0], [3, 3, 3], [0, 1, 0], [0, 0, 1]]);
  const types = new Map<number, string | null>([[10, "A"], [11, "A"], [12, "B"], [13, null], [14, "MN9"]]);

  it("keeps the top-N rows by total spikes, grouped by cell type, busiest group first", () => {
    const m = rasterModel(a, (r) => types.get(r) ?? null, { top: 3, emphasize: null });
    expect(m.rows.map((r) => r.row)).toEqual([12, 11, 10]); // B (9 spikes), then A (rows by activity)
    expect(m.groups).toEqual([
      { label: "B", start: 0, end: 1, emphasized: false },
      { label: "A", start: 1, end: 3, emphasized: false },
    ]);
  });

  it("always includes the emphasized (readout) rows even when quiet, groups them under the readout's name, and marks them", () => {
    const m = rasterModel(a, (r) => types.get(r) ?? null, { top: 2, emphasize: { rows: [14, 99], label: "MN9" } });
    expect(m.rows.map((r) => r.row)).toContain(14);
    const g = m.groups.find((x) => x.label === "MN9")!;
    expect(g.emphasized).toBe(true);
    expect(m.rows.find((r) => r.row === 14)!.emphasized).toBe(true);
  });

  it("lists the bins with spikes and their counts, plus the duration", () => {
    const m = rasterModel(a, (r) => types.get(r) ?? null, { top: 1, emphasize: null });
    expect(m.rows[0].spikes).toEqual([{ bin: 0, count: 3 }, { bin: 1, count: 3 }, { bin: 2, count: 3 }]);
    expect(m.durationMs).toBe(300);
    expect(m.binMs).toBe(100);
  });

  it("labels unannotated neurons as unknown type", () => {
    const m = rasterModel(a, (r) => types.get(r) ?? null, { top: 5, emphasize: null });
    expect(m.groups.some((g) => g.label === "unknown")).toBe(true);
  });
});

describe("groupLabelLayout", () => {
  it("centers a label on its group and drops the ones that would overlap the previous label", () => {
    const groups = [
      { label: "B", start: 0, end: 4, emphasized: false },
      { label: "A", start: 4, end: 5, emphasized: false },
      { label: "C", start: 5, end: 12, emphasized: false },
    ];
    const out = groupLabelLayout(groups, 4, 14);
    expect(out.map((l) => l.label)).toEqual(["B", "C"]); // A is one row tall, too close to B
    expect(out[0].y).toBe(8); // (0 + 4) / 2 rows * 4 px
  });
  it("never drops an emphasized label", () => {
    const groups = [
      { label: "B", start: 0, end: 4, emphasized: false },
      { label: "MN9", start: 4, end: 5, emphasized: true },
    ];
    expect(groupLabelLayout(groups, 4, 14).map((l) => l.label)).toEqual(["B", "MN9"]);
  });
});

describe("readout trace", () => {
  it("places points at bin centers", () => {
    const s = readoutSeries("baseline", [10, 20], 100, false);
    expect(s.points).toEqual([{ t: 50, hz: 10 }, { t: 150, hz: 20 }]);
    expect(s.dashed).toBe(false);
  });
  it("shares one nice y axis across series so they are directly comparable", () => {
    const m = traceModel([readoutSeries("baseline", [10, 61], 100, false), readoutSeries("silenced", [5, 3], 100, true)], 100, 200);
    expect(m.axis.max).toBe(80);
    expect(m.durationMs).toBe(200);
    expect(m.series.map((s) => s.dashed)).toEqual([false, true]);
  });
  it("builds an SVG path through the points", () => {
    const m = traceModel([readoutSeries("x", [0, 40], 100, false)], 100, 200);
    const d = tracePath(m.series[0], linearScale(0, 200, 0, 200), linearScale(0, m.axis.max, 100, 0));
    expect(d).toBe("M50,100 L150,0");
  });
  it("describes a bin for the tooltip with every series' exact value", () => {
    const m = traceModel([readoutSeries("baseline", [10, 61.25], 100, false), readoutSeries("silenced", [5, 3], 100, true)], 100, 200);
    expect(traceTooltip(m, 1)).toBe("t = 100–200 ms · baseline 61.3 Hz · silenced 3.0 Hz");
  });
  it("describes a raster cell", () => {
    expect(rasterTooltip({ label: "MN9", row: 7, spikes: [{ bin: 2, count: 4 }] }, 2, 100)).toBe("MN9 · row 7 · t = 200–300 ms · 4 spikes");
    expect(rasterTooltip({ label: "MN9", row: 7, spikes: [{ bin: 2, count: 1 }] }, 2, 100)).toBe("MN9 · row 7 · t = 200–300 ms · 1 spike");
  });
});

describe("parseRowTypes", () => {
  it("maps rows to cell types and tolerates junk entries", () => {
    const m = parseRowTypes([{ row: 3, cell_type: "MN9", super_class: "motor" }, { row: 4, cell_type: null }, { row: "x" }, null, { row: 5, cell_type: 7 }]);
    expect(Array.from(m.entries())).toEqual([[3, "MN9"], [4, null], [5, null]]);
  });
  it("returns an empty map for a non-array answer", () => {
    expect(parseRowTypes({ detail: "x" }).size).toBe(0);
  });
});

describe("smooth", () => {
  it("is a centered moving average that shrinks at the edges", () => {
    expect(smooth([0, 30, 0, 30, 0], 3)).toEqual([15, 10, 20, 10, 15]);
  });
  it("leaves the data alone for a window of 1 and never changes the length", () => {
    expect(smooth([1, 2, 3], 1)).toEqual([1, 2, 3]);
    expect(smooth([], 5)).toEqual([]);
    expect(smooth([4, 8], 9)).toHaveLength(2);
  });
  it("keeps the mean level of a constant trace", () => {
    expect(smooth([7, 7, 7, 7], 3)).toEqual([7, 7, 7, 7]);
  });
});

describe("smoothWindow", () => {
  it("covers about 25 ms with an odd number of bins, at least 1", () => {
    expect(smoothWindow(5)).toBe(5);
    expect(smoothWindow(10)).toBe(3);
    expect(smoothWindow(1)).toBe(25);
    expect(smoothWindow(100)).toBe(1);
  });
});
