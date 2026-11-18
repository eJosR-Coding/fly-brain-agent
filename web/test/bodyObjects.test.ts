import { describe, expect, it } from "vitest";
import { cubeBounds } from "../src/bodyCube";
import { MAX_OBJECTS, OBJECT_COLORS, validateObjects } from "../src/bodyObjects";

const b = cubeBounds(0, 50);
const sugar = { id: "s1", kind: "sugar", pos_mm: [5, 1, -3], size_mm: 2 };

describe("validateObjects (data in, nothing else)", () => {
  it("accepts a sugar and a coffee cube inside the glass cube", () => {
    const r = validateObjects([sugar, { id: "c1", kind: "coffee", pos_mm: [-5, 1.5, 4], size_mm: 3 }], b);
    expect(r.errors).toEqual([]);
    expect(r.objects).toEqual([
      { id: "s1", kind: "sugar", pos_mm: [5, 1, -3], size_mm: 2 },
      { id: "c1", kind: "coffee", pos_mm: [-5, 1.5, 4], size_mm: 3 },
    ]);
  });
  it("returns copies, not the caller's arrays", () => {
    const r = validateObjects([sugar], b);
    r.objects[0].pos_mm[0] = 99;
    expect(sugar.pos_mm[0]).toBe(5);
  });
  it("accepts an empty list (clears the layer)", () => {
    expect(validateObjects([], b)).toEqual({ objects: [], errors: [] });
  });
  it("rejects what is not a list", () => {
    expect(validateObjects("sugar", b).errors).toHaveLength(1);
    expect(validateObjects(null, b).objects).toEqual([]);
  });
  it.each([
    ["an unknown kind", { ...sugar, kind: "salt" }, /kind/],
    ["no id", { ...sugar, id: "" }, /id/],
    ["a non-string id", { ...sugar, id: 3 }, /id/],
    ["a short position", { ...sugar, pos_mm: [1, 2] }, /pos_mm/],
    ["a non-finite position", { ...sugar, pos_mm: [1, NaN, 2] }, /pos_mm/],
    ["a zero size", { ...sugar, size_mm: 0 }, /size_mm/],
    ["a negative size", { ...sugar, size_mm: -1 }, /size_mm/],
    ["a string size", { ...sugar, size_mm: "2" }, /size_mm/],
    ["a position outside the cube", { ...sugar, pos_mm: [30, 1, 0] }, /outside/],
    ["an object under the floor", { ...sugar, pos_mm: [0, 0, 0] }, /outside/],
    ["not an object", 7, /object/],
  ])("drops %s and says why, keeping the rest", (_n, bad, re) => {
    const r = validateObjects([bad, { ...sugar, id: "ok" }], b);
    expect(r.objects.map((o) => o.id)).toEqual(["ok"]);
    expect(r.errors).toHaveLength(1);
    expect(r.errors[0]).toMatch(re);
  });
  it("drops a repeated id", () => {
    const r = validateObjects([sugar, { ...sugar, pos_mm: [0, 1, 0] }], b);
    expect(r.objects).toHaveLength(1);
    expect(r.errors[0]).toMatch(/duplicate/);
  });
  it("caps the number of objects", () => {
    const many = Array.from({ length: MAX_OBJECTS + 5 }, (_, i) => ({ ...sugar, id: `s${i}` }));
    const r = validateObjects(many, b);
    expect(r.objects).toHaveLength(MAX_OBJECTS);
    expect(r.errors.some((e) => /at most/.test(e))).toBe(true);
  });
  it("sugar reads white-ish and coffee dark brown", () => {
    const lum = (c: number) => ((c >> 16) & 255) + ((c >> 8) & 255) + (c & 255);
    expect(lum(OBJECT_COLORS.sugar)).toBeGreaterThan(600);
    expect(lum(OBJECT_COLORS.coffee)).toBeLessThan(250);
    expect((OBJECT_COLORS.coffee >> 16) & 255).toBeGreaterThan(OBJECT_COLORS.coffee & 255); // brown, not grey
  });
});
