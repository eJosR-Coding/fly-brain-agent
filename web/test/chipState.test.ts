import { describe, expect, it } from "vitest";
import { chartsChipState, simChipsDisabled } from "../src/chipState";

describe("chartsChipState", () => {
  it("is disabled and inactive with nothing to chart", () => {
    expect(chartsChipState({ hasContent: false, panelOpen: false })).toEqual({ disabled: true, active: false });
  });
  it("is enabled once there is a recording or a chart", () => {
    expect(chartsChipState({ hasContent: true, panelOpen: false })).toEqual({ disabled: false, active: false });
  });
  it("is active whenever the panel is open, even if the chip was built after the panel opened", () => {
    expect(chartsChipState({ hasContent: true, panelOpen: true })).toEqual({ disabled: false, active: true });
  });
  it("an open panel without content never enables the chip (invalid combination stays disabled)", () => {
    expect(chartsChipState({ hasContent: false, panelOpen: true }).disabled).toBe(true);
  });
});

describe("simChipsDisabled", () => {
  const base = { simulating: false, streaming: false, simulationOff: false };
  it("is enabled when idle", () => expect(simChipsDisabled(base)).toBe(false));
  it("is disabled while a simulation runs", () => expect(simChipsDisabled({ ...base, simulating: true })).toBe(true));
  it("is disabled while the chat streams", () => expect(simChipsDisabled({ ...base, streaming: true })).toBe(true));
  it("is disabled when the server has no simulator", () => expect(simChipsDisabled({ ...base, simulationOff: true })).toBe(true));
  it("finishing a run does not re-enable chips on a server without simulator", () => {
    expect(simChipsDisabled({ simulating: false, streaming: false, simulationOff: true })).toBe(true);
  });
});
