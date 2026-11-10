import { describe, expect, it } from "vitest";
import { createOrbitGate, orbitDefault } from "../src/orbit";

describe("orbitDefault", () => {
  it("is on unless the user prefers reduced motion", () => {
    expect(orbitDefault(false)).toBe(true);
    expect(orbitDefault(true)).toBe(false);
  });
});

describe("createOrbitGate", () => {
  it("spins when enabled and idle", () => {
    expect(createOrbitGate(true, 3000).active(0)).toBe(true);
  });
  it("never spins when disabled", () => {
    const g = createOrbitGate(false, 3000);
    expect(g.active(0)).toBe(false);
    g.setEnabled(true);
    expect(g.active(0)).toBe(true);
  });
  it("pauses while the pointer is down and for the resume delay after release", () => {
    const g = createOrbitGate(true, 3000);
    g.press();
    expect(g.active(10_000)).toBe(false);
    g.release(10_000);
    expect(g.active(10_000)).toBe(false);
    expect(g.active(12_999)).toBe(false);
    expect(g.active(13_000)).toBe(true);
  });
  it("a touch (camera preset, wheel) pauses for the delay too", () => {
    const g = createOrbitGate(true, 2000);
    g.touch(500);
    expect(g.active(2_499)).toBe(false);
    expect(g.active(2_500)).toBe(true);
  });
  it("disabling wins over everything", () => {
    const g = createOrbitGate(true, 100);
    g.setEnabled(false);
    expect(g.active(1e9)).toBe(false);
    expect(g.enabled).toBe(false);
  });
});
