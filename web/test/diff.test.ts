import { describe, expect, it } from "vitest";
import { decodeFrames, type Activity } from "../src/activity";
import { computeDiff, diffBrightness, diffLegend, diffShapes, meanHz, rateHz, DIFF_KIND, SHAPE } from "../src/diff";

function wire(rows: number[], counts: number[][]): ArrayBuffer {
  const nBins = counts[0].length;
  const buf = new ArrayBuffer(rows.length * 4 + rows.length * nBins);
  const view = new DataView(buf);
  rows.forEach((r, i) => view.setUint32(i * 4, r, true));
  new Uint8Array(buf, rows.length * 4).set(counts.flat());
  return buf;
}
/** 4 bins of 250 ms = 1 s; `trials` scales the rate. */
const act = (rows: number[], counts: number[][], trials = 1): Activity => ({
  ...decodeFrames(wire(rows, counts), rows.length, counts[0].length, 250),
  trials,
});

describe("rateHz", () => {
  it("is spikes per trial per second", () => {
    const a = act([5], [[2, 0, 4, 0]], 3); // 6 spikes over 3 trials of 1 s
    expect(rateHz(a, 0)).toBeCloseTo(2, 6);
  });
});

describe("computeDiff", () => {
  const base = act([1, 2, 3], [[5, 5, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]].map((r, i) => (i === 2 ? [1, 0, 0, 0] : r)));
  const sil = act([2, 3, 4], [[0, 0, 0, 0], [6, 6, 0, 0], [3, 0, 0, 0]].map((r, i) => (i === 0 ? [1, 0, 0, 0] : r)));
  // rates (Hz): row1 base 10 sil -   | row2 base 2 sil 1 | row3 base 1 sil 12 | row4 base - sil 3

  it("covers the union of rows, ascending, with silenced - baseline in Hz", () => {
    const d = computeDiff(base, sil, [], 0);
    expect(Array.from(d.rows)).toEqual([1, 2, 3, 4]);
    expect(Array.from(d.delta)).toEqual([-10, -1, 11, 3]);
  });

  it("classifies lost, gained and unchanged against a threshold", () => {
    const d = computeDiff(base, sil, [], 1.5);
    expect(Array.from(d.kind)).toEqual([DIFF_KIND.lost, DIFF_KIND.unchanged, DIFF_KIND.gained, DIFF_KIND.gained]);
    expect(d.lost).toBe(1);
    expect(d.gained).toBe(2);
  });

  it("marks the silenced neurons and keeps them out of the lost/gained counts", () => {
    const d = computeDiff(base, sil, [1, 9], 1.5);
    expect(Array.from(d.rows)).toEqual([1, 2, 3, 4, 9]); // a silenced neuron that never spiked is still drawn
    expect(d.kind[0]).toBe(DIFF_KIND.silenced);
    expect(d.kind[4]).toBe(DIFF_KIND.silenced);
    expect(d.lost).toBe(0);
    expect(d.gained).toBe(2);
  });

  it("refuses recordings with different time bases", () => {
    expect(() => computeDiff(base, { ...sil, nBins: 2 }, [], 0)).toThrow(/time base|bins/);
  });

  it("tracks the largest change", () => {
    expect(computeDiff(base, sil, [], 0).maxAbs).toBe(11);
  });
});

describe("diffBrightness", () => {
  it("hides unchanged, scales the rest with the magnitude, keeps silenced marks at a fixed grey", () => {
    const base = act([1, 2, 3, 4], [[10, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]);
    const sil = act([1, 2, 3, 4], [[0, 0, 0, 0], [8, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]);
    const d = computeDiff(base, sil, [4], 1);
    const out = new Float32Array(d.rows.length);
    diffBrightness(d, out);
    expect(out[2]).toBe(0); // unchanged
    expect(out[0]).toBe(1); // the biggest change
    expect(out[1]).toBeGreaterThan(0.35);
    expect(out[1]).toBeLessThan(out[0]);
    expect(out[3]).toBeGreaterThan(0);
    expect(out[3]).toBeLessThan(1); // the silenced mark is grey
  });
});

describe("legend and readout", () => {
  it("meanHz averages the per-bin trace", () => {
    expect(meanHz([10, 20, 30, 40])).toBe(25);
    expect(meanHz([])).toBe(0);
  });
  it("summarizes lost and gained, with the readout when known", () => {
    expect(diffLegend(37, 4, null)).toBe("diff · 37 lost · 4 gained");
    expect(diffLegend(1234, 4, { label: "MN9", before: 61.2, after: 57.6 })).toBe("diff · 1,234 lost · 4 gained · MN9 61 → 58 Hz");
  });
});

describe("diffShapes", () => {
  it("uses shape, not color: lost = ring, gained = filled dot, silenced = cross", () => {
    const base = act([1, 2, 3], [[10, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]);
    const sil = act([1, 2, 3], [[0, 0, 0, 0], [9, 0, 0, 0], [1, 0, 0, 0]]);
    const d = computeDiff(base, sil, [3], 1);
    const out = new Float32Array(3);
    diffShapes(d, out);
    expect(Array.from(out)).toEqual([SHAPE.ring, SHAPE.dot, SHAPE.cross]);
  });
});
