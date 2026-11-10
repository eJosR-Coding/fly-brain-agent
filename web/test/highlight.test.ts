import { describe, expect, it } from "vitest";
import { writeSelectionIndex } from "../src/highlight";

describe("writeSelectionIndex", () => {
  it("writes unique in-range indices ascending and returns how many", () => {
    const target = new Uint32Array(10);
    const n = writeSelectionIndex(target, [7, 2, 2, 5, 7]);
    expect(n).toBe(3);
    expect(Array.from(target.subarray(0, n))).toEqual([2, 5, 7]);
  });

  it("ignores negative, fractional and out-of-range values", () => {
    const target = new Uint32Array(4);
    const n = writeSelectionIndex(target, [-1, 0, 1.5, 3, 4, 99]);
    expect(Array.from(target.subarray(0, n))).toEqual([0, 3]);
  });

  it("an empty selection draws nothing", () => {
    expect(writeSelectionIndex(new Uint32Array(5), [])).toBe(0);
  });

  it("accepts typed arrays and handles a selection of every point", () => {
    const target = new Uint32Array(6);
    const n = writeSelectionIndex(target, Uint32Array.from([5, 4, 3, 2, 1, 0]));
    expect(n).toBe(6);
    expect(Array.from(target)).toEqual([0, 1, 2, 3, 4, 5]);
  });
});
