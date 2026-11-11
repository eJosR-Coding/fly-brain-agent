import { describe, expect, it } from "vitest";
import {
  doseModel, doseTooltip, formatExperimentProgress, logScale, parseChart, parseProgress, screenModel, screenTooltip, symmetricTicks,
} from "../src/experiments";

const dose = {
  kind: "dose_response", id: "chart_a", readout_label: "MN9", stimulus_label: "SUGAR", trials: 3, duration_ms: 300, ec50_hz: 77.5,
  points: [
    { rate_hz: 10, mean_hz: 0, ci_low: 0, ci_high: 0 },
    { rate_hz: 50, mean_hz: 17.8, ci_low: 0, ci_high: 49.1 },
    { rate_hz: 200, mean_hz: 85.6, ci_low: 72.9, ci_high: 98.2 },
  ],
};
const screen = {
  kind: "screen", id: "chart_b", readout_label: "MN9", stimulus_label: "SUGAR", trials: 3, duration_ms: 300, baseline_simulation_id: "sim_b",
  rows: [
    { label: "T4a", row: 5, delta_hz: -30.2, ci_low: -45, ci_high: -15.4, p: 0.001, significant: true, simulation_id: "sim_1" },
    { label: "CB0118", row: 9, delta_hz: 12.2, ci_low: -9.5, ci_high: 34, p: 0.19, significant: false, simulation_id: "sim_2" },
  ],
};

describe("parseChart", () => {
  it("accepts both kinds and rejects malformed payloads with a reason", () => {
    expect(typeof parseChart(dose)).toBe("object");
    expect(typeof parseChart(screen)).toBe("object");
    expect(parseChart(null)).toMatch(/not an object/);
    expect(parseChart({ ...dose, kind: "pie" })).toMatch(/kind/);
    expect(parseChart({ ...dose, points: "no" })).toMatch(/points/);
    expect(parseChart({ ...dose, points: [{ rate_hz: 0, mean_hz: 1, ci_low: 0, ci_high: 1 }] })).toMatch(/point/);
    expect(parseChart({ ...screen, rows: [{ label: 3 }] })).toMatch(/row/);
  });
  it("keeps a missing interval as null (one trial) and ec50 as null", () => {
    const c = parseChart({ ...dose, ec50_hz: undefined, points: [{ rate_hz: 10, mean_hz: 2, ci_low: null, ci_high: null }] });
    expect(c).toMatchObject({ ec50: null, points: [{ low: null, high: null }] });
  });
});

describe("logScale", () => {
  it("maps decades evenly and clamps", () => {
    const s = logScale(10, 1000, 0, 200);
    expect(s(10)).toBe(0);
    expect(s(100)).toBeCloseTo(100);
    expect(s(1000)).toBe(200);
    expect(s(1)).toBe(0);
  });
  it("survives one point", () => {
    expect(logScale(50, 50, 5, 9)(50)).toBe(5);
  });
});

describe("doseModel", () => {
  const c = parseChart(dose);
  if (typeof c === "string" || c.kind !== "dose_response") throw new Error("setup");
  it("puts rates on a log x axis, ticks at the tested rates, and keeps the y axis at zero", () => {
    const m = doseModel(c, { left: 40, right: 400, top: 10, bottom: 130 });
    expect(m.points.map((p) => p.rate)).toEqual([10, 50, 200]);
    expect(m.points[0].x).toBe(40);
    expect(m.points[2].x).toBe(400);
    expect(m.points[1].x).toBeGreaterThan(40 + (400 - 40) / 2 - 1); // 50 Hz sits past the log midpoint of 10..200? (log 50 / log 200)
    expect(m.xTicks.map((t) => t.rate)).toEqual([10, 50, 200]);
    expect(m.yAxis.ticks[0]).toBe(0);
    expect(m.yAxis.max).toBeGreaterThanOrEqual(98.2); // the CI fits
    expect(m.points[2].yHigh).toBeLessThan(m.points[2].y);
  });
  it("marks the half-max rate with a position and a label", () => {
    const m = doseModel(c, { left: 40, right: 400, top: 10, bottom: 130 });
    expect(m.ec50?.label).toBe("EC50 77.5 Hz");
    expect(m.ec50!.x).toBeGreaterThan(m.points[1].x);
    expect(m.ec50!.x).toBeLessThan(m.points[2].x);
  });
  it("has no marker without an EC50", () => {
    const none = parseChart({ ...dose, ec50_hz: null }) as typeof c;
    expect(doseModel(none, { left: 0, right: 100, top: 0, bottom: 50 }).ec50).toBeNull();
  });
  it("builds a CI band polygon from the intervals", () => {
    const m = doseModel(c, { left: 40, right: 400, top: 10, bottom: 130 });
    expect(m.band).toMatch(/^M/);
    expect(m.band.split("L").length).toBe(2 * m.points.length);
  });
  it("tooltips carry the exact values", () => {
    expect(doseTooltip(c.points[1], "MN9")).toBe("50 Hz stimulus · MN9 17.8 Hz (95% CI 0.0–49.1)");
    expect(doseTooltip({ rate: 5, mean: 2, low: null, high: null }, "MN9")).toBe("5 Hz stimulus · MN9 2.0 Hz (no interval: 1 trial)");
  });
});

describe("symmetricTicks", () => {
  it("covers the extent around zero with a tick at 0", () => {
    const t = symmetricTicks(-45, 34);
    expect(t.ticks).toContain(0);
    expect(t.min).toBeLessThanOrEqual(-45);
    expect(t.max).toBeGreaterThanOrEqual(34);
  });
  it("survives an all-zero extent", () => {
    const t = symmetricTicks(0, 0);
    expect(t.max).toBeGreaterThan(t.min);
  });
});

describe("screenModel", () => {
  const c = parseChart(screen);
  if (typeof c === "string" || c.kind !== "screen") throw new Error("setup");
  const box = { left: 80, right: 380, top: 6, rowPx: 16 };
  it("lists rows top to bottom in the given order, with whiskers and a zero line", () => {
    const m = screenModel(c, box);
    expect(m.rows.map((r) => r.label)).toEqual(["T4a", "CB0118"]);
    expect(m.rows[0].y).toBeLessThan(m.rows[1].y);
    for (const r of m.rows) {
      expect(r.xLow).toBeLessThanOrEqual(r.x);
      expect(r.x).toBeLessThanOrEqual(r.xHigh);
    }
    expect(m.zeroX).toBeGreaterThan(80);
    expect(m.zeroX).toBeLessThan(380);
    expect(m.rows[0].x).toBeLessThan(m.zeroX); // negative delta is left of zero
    expect(m.height).toBe(6 + 2 * 16 + 40);
  });
  it("fills the significant ones and leaves the others hollow", () => {
    expect(screenModel(c, box).rows.map((r) => r.filled)).toEqual([true, false]);
  });
  it("tooltips name the verdict in words", () => {
    expect(screenTooltip(c.rows[0])).toBe("T4a · row 5 · Δ -30.2 Hz (95% CI -45.0 – -15.4) · p 0.001 · significant after correction · click to open its map");
    expect(screenTooltip(c.rows[1])).toContain("not significant");
    expect(screenTooltip({ ...c.rows[1], low: null, high: null, p: null })).toContain("no test");
  });
});

describe("progress", () => {
  it("parses and formats the live label", () => {
    expect(parseProgress({ experiment: "silencing_screen", done: 3, total: 8 })).toEqual({ experiment: "silencing_screen", done: 3, total: 8 });
    expect(parseProgress({ experiment: "x", done: -1, total: 8 })).toBeNull();
    expect(parseProgress({ experiment: "silencing_screen", done: 9, total: 8 })).toBeNull();
    expect(formatExperimentProgress({ experiment: "silencing_screen", done: 3, total: 8 })).toBe("screen · 3/8");
    expect(formatExperimentProgress({ experiment: "dose_response", done: 0, total: 5 })).toBe("dose · 0/5");
    expect(formatExperimentProgress({ experiment: "other", done: 1, total: 2 })).toBe("other · 1/2");
  });
});
