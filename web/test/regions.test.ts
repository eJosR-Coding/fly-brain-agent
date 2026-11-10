import { describe, expect, it } from "vitest";
import { buildRegionColors, hexToRgb, loadLandmarks, loadRegions, regionPalette } from "../src/regions";

const TONES = { central: "#ff0000", optic: "#00ff00", sensory: "#0000ff", other: "#808080" };
const TABLE = { "0": "central", "1": "optic", "2": "sensory", "3": "other" };

describe("hexToRgb", () => {
  it("parses #rrggbb and #rgb into 0..1", () => {
    expect(hexToRgb("#ff8000")).toEqual([1, 128 / 255, 0]);
    expect(hexToRgb("#0f0")).toEqual([0, 1, 0]);
  });
  it("falls back to mid grey on garbage instead of NaN", () => {
    expect(hexToRgb("banana")).toEqual([0.5, 0.5, 0.5]);
  });
});

describe("regionPalette", () => {
  it("maps each code of the meta table to its tone", () => {
    const p = regionPalette(TABLE, TONES, "#111111");
    expect(Array.from(p.slice(0, 3))).toEqual([1, 0, 0]);
    expect(Array.from(p.slice(3, 6))).toEqual([0, 1, 0]);
    expect(p.length).toBe(4 * 3);
  });
  it("uses the fallback for names without a tone and for unknown codes", () => {
    const p = regionPalette({ "0": "central", "5": "mystery" }, TONES, "#ffffff");
    expect(Array.from(p.slice(5 * 3, 5 * 3 + 3))).toEqual([1, 1, 1]);
    expect(Array.from(p.slice(3, 6))).toEqual([1, 1, 1]); // code 1 is absent
  });
  it("ignores keys that are not small non-negative integers", () => {
    const p = regionPalette({ "-1": "central", x: "optic", "0": "central" }, TONES, "#ffffff");
    expect(p.length).toBe(3);
  });
});

describe("buildRegionColors", () => {
  it("writes one rgb per point from the palette", () => {
    const p = regionPalette(TABLE, TONES, "#111111");
    const c = buildRegionColors(new Uint8Array([0, 2, 1]), p);
    expect(Array.from(c)).toEqual([1, 0, 0, 0, 0, 1, 0, 1, 0]);
  });
  it("a code outside the palette gets the last palette entry, never undefined/NaN", () => {
    const p = regionPalette(TABLE, TONES, "#111111");
    const c = buildRegionColors(new Uint8Array([200]), p);
    expect(Array.from(c).every(Number.isFinite)).toBe(true);
  });
});

function fakeFetch(map: Record<string, Response>) {
  return (async (url: string) => map[url] ?? new Response("nope", { status: 404 })) as unknown as typeof fetch;
}

describe("loadRegions", () => {
  it("returns the uint8 codes and the table", async () => {
    const f = fakeFetch({ "/api/brain/regions": new Response(new Uint8Array([0, 1, 2]).buffer) });
    const r = await loadRegions(3, TABLE, f);
    expect(Array.from(r.codes)).toEqual([0, 1, 2]);
    expect(r.table).toEqual(TABLE);
  });
  it("rejects when the length is not the point count", async () => {
    const f = fakeFetch({ "/api/brain/regions": new Response(new Uint8Array([0, 1]).buffer) });
    await expect(loadRegions(3, TABLE, f)).rejects.toThrow(/2 bytes.*3 points/);
  });
  it("rejects on http errors", async () => {
    await expect(loadRegions(3, TABLE, fakeFetch({}))).rejects.toThrow(/404/);
  });
});

describe("loadLandmarks", () => {
  const good = {
    landmarks: [{ name: "central brain", x: 1, y: 2, z: 3, count: 5 }],
    sides: { left: { lobe_center: [1, 2, 3], radius: 4, lamina_direction: [1, 0, 0] } },
  };
  it("accepts the contract", async () => {
    const f = fakeFetch({ "/api/brain/landmarks": new Response(JSON.stringify(good)) });
    expect((await loadLandmarks(f)).landmarks[0].name).toBe("central brain");
  });
  it("rejects a malformed payload with a clear message", async () => {
    const f = fakeFetch({ "/api/brain/landmarks": new Response(JSON.stringify({ landmarks: "x" })) });
    await expect(loadLandmarks(f)).rejects.toThrow(/landmarks/);
  });
  it("drops sides/landmarks with non-finite numbers instead of passing NaN on", async () => {
    const bad = {
      landmarks: [{ name: "a", x: "1", y: 2, z: 3, count: 1 }, good.landmarks[0]],
      sides: { left: { lobe_center: [1, 2, null], radius: 4, lamina_direction: null }, right: good.sides.left },
    };
    const f = fakeFetch({ "/api/brain/landmarks": new Response(JSON.stringify(bad)) });
    const lm = await loadLandmarks(f);
    expect(lm.landmarks.map((m) => m.name)).toEqual(["central brain"]);
    expect(Object.keys(lm.sides)).toEqual(["right"]);
  });
});
