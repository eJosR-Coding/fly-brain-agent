import { describe, expect, it } from "vitest";
import { keyRows, loadNt, ntPalette, ntWeights, NT_TIERS, ntTier } from "../src/transmitters";
import { buildRegionColors } from "../src/regions";

const TABLE = { "0": "acetylcholine", "1": "glutamate", "2": "gaba", "3": "dopamine", "4": "serotonin", "5": "octopamine", "6": "unknown" };

describe("ntTier", () => {
  it("rare transmitters are brightest, then octopamine, gaba, glutamate, acetylcholine, unknown", () => {
    const t = (n: string) => ntTier(n).tone;
    expect(t("dopamine")).toBeGreaterThanOrEqual(t("serotonin"));
    expect(t("serotonin")).toBeGreaterThan(t("octopamine"));
    expect(t("octopamine")).toBeGreaterThan(t("gaba"));
    expect(t("gaba")).toBeGreaterThan(t("glutamate"));
    expect(t("glutamate")).toBeGreaterThan(t("acetylcholine"));
    expect(t("acetylcholine")).toBeGreaterThan(t("unknown"));
  });
  it("tiers are far enough apart to read, and stay in [0, 1]", () => {
    const tones = Object.values(NT_TIERS).map((x) => x.tone).sort((a, b) => b - a);
    for (const v of tones) { expect(v).toBeGreaterThan(0); expect(v).toBeLessThanOrEqual(1); }
    for (let i = 2; i < tones.length; i++) expect(tones[i - 1] - tones[i]).toBeGreaterThanOrEqual(0.12);
  });
  it("an unseen name falls back to the unknown tier", () => {
    expect(ntTier("histamine")).toEqual(ntTier("unknown"));
  });
});

describe("ntPalette / ntWeights", () => {
  const { palette, sizes } = ntPalette(TABLE);
  it("greys only: strictly black and white", () => {
    expect(palette.length).toBe(7 * 3);
    for (let c = 0; c < 7; c++) {
      expect(palette[c * 3]).toBeCloseTo(palette[c * 3 + 1]);
      expect(palette[c * 3]).toBeCloseTo(palette[c * 3 + 2]);
    }
    expect(palette[3 * 3]).toBeGreaterThan(palette[0]); // dopamine brighter than acetylcholine
  });
  it("colors and sizes follow the codes; an out-of-table code gets the last entry", () => {
    const codes = new Uint8Array([3, 0, 99]);
    const rgb = buildRegionColors(codes, palette);
    expect(rgb[0]).toBeCloseTo(palette[9]);
    expect(rgb[3]).toBeCloseTo(palette[0]);
    expect(rgb[6]).toBeCloseTo(palette[18]);
    const w = ntWeights(codes, sizes);
    expect(Array.from(w)).toEqual([sizes[3], sizes[0], sizes[6]].map((v) => Math.fround(v)));
    expect(sizes[3]).toBeGreaterThan(sizes[0]); // rare ones are a bit larger
  });
});

describe("keyRows", () => {
  const swatch = (n: string) => `#${n.length}`;
  it("counts per code, in table order, skipping empty codes", () => {
    const rows = keyRows(new Uint8Array([0, 0, 0, 3, 6, 6]), TABLE, swatch);
    expect(rows.map((r) => [r.name, r.count])).toEqual([["acetylcholine", 3], ["dopamine", 1], ["unknown", 2]]);
    expect(rows[0].hex).toBe("#13");
  });
  it("ignores codes outside the table", () => {
    expect(keyRows(new Uint8Array([0, 200]), TABLE, swatch).map((r) => r.count)).toEqual([1]);
  });
});

describe("loadNt", () => {
  const ok = (bytes: number[]) => (async () => new Response(new Uint8Array(bytes))) as unknown as typeof fetch;
  it("reads one uint8 per point", async () => {
    const nt = await loadNt(3, TABLE, ok([0, 3, 6]));
    expect(Array.from(nt.codes)).toEqual([0, 3, 6]);
    expect(nt.table).toBe(TABLE);
  });
  it("rejects a length that does not match the point count", async () => {
    await expect(loadNt(5, TABLE, ok([0, 3, 6]))).rejects.toThrow(/does not match 5 points/);
  });
  it("rejects HTTP errors", async () => {
    await expect(loadNt(1, TABLE, (async () => new Response("no", { status: 404 })) as unknown as typeof fetch)).rejects.toThrow(/HTTP 404/);
  });
});
