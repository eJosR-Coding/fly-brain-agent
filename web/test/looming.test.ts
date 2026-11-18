import { describe, expect, it } from "vitest";
import { LOOMING_RUN, loomingLegend, parseLoomingRun } from "../src/looming";

const result = {
  experiment: "looming_escape", input: "motion", simulation_id: "sim_loom", runtime_s: 46,
  stimulus: { duration_ms: 500, trials: 5, side: "right", input: "motion" },
  conditions: { looming: { dnp01_hz: 9.4, simulation_id: "sim_loom" }, receding: { dnp01_hz: 0.8 }, dimming: { dnp01_hz: 0 }, control: { dnp01_hz: 3.8 } },
  verdict: "model prediction: looming drives more DNp01 than receding",
};

describe("looming chip", () => {
  it("requests the validated configuration: motion input, right eye, 300 Hz, both polarities", () => {
    expect(LOOMING_RUN).toEqual({ input: "motion", side: "right", rate_hz: 300, polarity: "both", trials: 5, duration_ms: 500, steps: 10 });
  });

  it("plays the LOOMING condition's recording and keeps the other conditions' DNp01 for the caption", () => {
    const r = parseLoomingRun(result);
    expect(typeof r).not.toBe("string");
    if (typeof r === "string") return;
    expect(r.simulationId).toBe("sim_loom");
    expect(r.durationMs).toBe(500);
    expect(r.dnp01).toEqual({ looming: 9.4, receding: 0.8, dimming: 0, control: 3.8 });
  });

  it("refuses an answer that is not a looming run", () => {
    expect(parseLoomingRun(null)).toBe("not an object");
    expect(parseLoomingRun({ ...result, simulation_id: "", conditions: { ...result.conditions, looming: { dnp01_hz: 9.4 } } })).toMatch(/simulation_id/);
    expect(parseLoomingRun({ ...result, conditions: {} })).toMatch(/looming/);
    expect(parseLoomingRun({ ...result, conditions: { looming: { dnp01_hz: "x" } } })).toMatch(/dnp01_hz/);
  });

  it("labels the legend as a model prediction with the four DNp01 rates", () => {
    const r = parseLoomingRun(result);
    if (typeof r === "string") throw new Error(r);
    const text = loomingLegend(r);
    expect(text).toContain("model");
    expect(text).toContain("DNp01");
    expect(text).toMatch(/looming 9\.4 Hz/);
    expect(text).toMatch(/receding 0\.8 Hz/);
    expect(text).toMatch(/dimming 0 Hz/);
    expect(text).toMatch(/control 3\.8 Hz/);
  });
});
