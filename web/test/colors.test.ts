import { describe, expect, it } from "vitest";
import { applyHighlight, baseColors, clearHighlight } from "../src/colors";

const GREY: [number, number, number] = [0.5, 0.5, 0.5];
const ACCENT: [number, number, number] = [1, 0.25, 0];

describe("colors", () => {
  it("builds a base array with one rgb triple per point", () => {
    const c = baseColors(3, GREY);
    expect(Array.from(c)).toEqual([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
  });

  it("recolors exactly the given indices", () => {
    const c = baseColors(4, GREY);
    applyHighlight(c, Uint32Array.from([1, 3]), ACCENT);
    expect(Array.from(c)).toEqual([0.5, 0.5, 0.5, 1, 0.25, 0, 0.5, 0.5, 0.5, 1, 0.25, 0]);
  });

  it("ignores out-of-range indices", () => {
    const c = baseColors(2, GREY);
    applyHighlight(c, [5, 2, 1], ACCENT);
    expect(Array.from(c)).toEqual([0.5, 0.5, 0.5, 1, 0.25, 0]);
  });

  it("clear restores the base in place", () => {
    const base = baseColors(3, GREY);
    const c = base.slice();
    applyHighlight(c, [0, 2], ACCENT);
    const same = clearHighlight(c, base);
    expect(same).toBe(c);
    expect(Array.from(c)).toEqual(Array.from(base));
  });
});
