import { describe, expect, it } from "vitest";
import { modeLegend, viewPlan } from "../src/modes";

describe("viewPlan", () => {
  it("animates the wave unless motion is reduced, in which case it shows the summed still", () => {
    expect(viewPlan("wave", { hasDiff: false, reducedMotion: false })).toMatchObject({
      mode: "wave", rows: "activity", animate: true, scale: "none", staticView: false, timeLabel: "playhead", shapes: false,
    });
    expect(viewPlan("wave", { hasDiff: true, reducedMotion: true })).toMatchObject({
      mode: "wave", animate: false, timeLabel: "summed",
    });
  });

  it("latency is a static map with its own scale on the recorded rows", () => {
    expect(viewPlan("latency", { hasDiff: false, reducedMotion: false })).toMatchObject({
      mode: "latency", rows: "activity", animate: false, scale: "latency", staticView: true, timeLabel: "none", shapes: false,
    });
  });

  it("diff draws the union rows with shapes and its key, only when a baseline exists", () => {
    expect(viewPlan("diff", { hasDiff: true, reducedMotion: false })).toMatchObject({
      mode: "diff", rows: "diff", animate: false, scale: "diffKey", staticView: true, timeLabel: "none", shapes: true,
    });
  });

  it("asking for the diff without a baseline falls back to the wave", () => {
    expect(viewPlan("diff", { hasDiff: false, reducedMotion: false }).mode).toBe("wave");
  });
});

describe("modeLegend", () => {
  it("keeps the wave legend (null), names latency, and summarizes the diff", () => {
    expect(modeLegend("wave", { nRows: 10, diff: null, readout: null })).toBeNull();
    expect(modeLegend("latency", { nRows: 12345, diff: null, readout: null })).toBe("latency · 12,345 neurons spiked");
    const text = modeLegend("diff", { nRows: 1, diff: { lost: 3, gained: 4 }, readout: { label: "MN9", before: 60, after: 71 } });
    expect(text).toContain("3");
    expect(text).toContain("MN9");
  });
  it("falls back to null for a diff legend without a diff", () => {
    expect(modeLegend("diff", { nRows: 1, diff: null, readout: null })).toBeNull();
  });
});
