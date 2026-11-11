import { describe, expect, it } from "vitest";
import { decodeFrames, type Activity } from "../src/activity";
import {
  fillLatency, firstSpikeMs, latencyGradientStops, latencyLegend, latencyLevel, latencyTicks,
} from "../src/latency";
import { fallbackMode, modesFor } from "../src/modes";

function wire(rows: number[], counts: number[][]): ArrayBuffer {
  const nBins = counts[0].length;
  const buf = new ArrayBuffer(rows.length * 4 + rows.length * nBins);
  const view = new DataView(buf);
  rows.forEach((r, i) => view.setUint32(i * 4, r, true));
  new Uint8Array(buf, rows.length * 4).set(counts.flat());
  return buf;
}
const act = (counts: number[][], binMs = 10): Activity =>
  decodeFrames(wire(counts.map((_, i) => 10 + i), counts), counts.length, counts[0].length, binMs);

describe("firstSpikeMs", () => {
  it("is the first non-zero bin times the bin width, per row", () => {
    const a = act([[0, 0, 3, 1], [1, 0, 0, 0], [0, 4, 0, 2]]);
    expect(Array.from(firstSpikeMs(a))).toEqual([20, 0, 10]);
  });
  it("is NaN for a row without any spike", () => {
    expect(firstSpikeMs(act([[0, 0, 0]]))[0]).toBeNaN();
  });
});

describe("latencyLevel", () => {
  it("is monotonic: early is bright, late is dim but still visible", () => {
    expect(latencyLevel(0, 1000)).toBe(1);
    expect(latencyLevel(250, 1000)).toBeGreaterThan(latencyLevel(500, 1000));
    expect(latencyLevel(500, 1000)).toBeGreaterThan(latencyLevel(1000, 1000));
    expect(latencyLevel(1000, 1000)).toBeGreaterThanOrEqual(0.15);
  });
  it("clamps outside the range and hides NaN", () => {
    expect(latencyLevel(-5, 1000)).toBe(1);
    expect(latencyLevel(5000, 1000)).toBe(latencyLevel(1000, 1000));
    expect(latencyLevel(NaN, 1000)).toBe(0);
  });
});

describe("fillLatency", () => {
  it("writes one level per row", () => {
    const a = act([[1, 0, 0, 0], [0, 0, 0, 1]], 250); // 1000 ms
    const out = new Float32Array(2);
    fillLatency(a, out);
    expect(out[0]).toBe(1);
    expect(out[1]).toBeCloseTo(latencyLevel(750, 1000), 5);
    expect(out[0]).toBeGreaterThan(out[1]);
  });
  it("refuses a too-small buffer", () => {
    expect(() => fillLatency(act([[1], [1]]), new Float32Array(1))).toThrow(/buffer/);
  });
});

describe("latency legend", () => {
  it("labels the range and puts ticks at both ends and the middle", () => {
    expect(latencyLegend(1000)).toBe("first spike · 0 ms → 1000 ms");
    expect(latencyTicks(1000)).toEqual([
      { frac: 0, label: "0 ms" }, { frac: 0.5, label: "500 ms" }, { frac: 1, label: "1000 ms" },
    ]);
  });
  it("gradient stops go from white to the late grey, matching the neuron levels", () => {
    const [first, last] = [latencyGradientStops()[0], latencyGradientStops().at(-1)!];
    expect(first).toEqual({ frac: 0, grey: 255 });
    expect(last.frac).toBe(1);
    expect(last.grey).toBe(Math.round(255 * latencyLevel(1000, 1000)));
  });
});

describe("view modes", () => {
  it("offers diff only when there is a comparison", () => {
    expect(modesFor(false)).toEqual(["wave", "latency"]);
    expect(modesFor(true)).toEqual(["wave", "latency", "diff"]);
  });
  it("falls back to the wave when the current mode is no longer offered", () => {
    expect(fallbackMode("diff", false)).toBe("wave");
    expect(fallbackMode("diff", true)).toBe("diff");
    expect(fallbackMode("latency", false)).toBe("latency");
  });
});
