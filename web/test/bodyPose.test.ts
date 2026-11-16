import { describe, expect, it } from "vitest";
import { MAPPING, captionFor, poseFor, sampleReadout } from "../src/bodyPose";

const FORBIDDEN = /\b(die|dies|dead|death|overdose|poison|kill|suffer|pain|feel)\b/i;

describe("MAPPING", () => {
  it("mn9 drives the proboscis, validated, fixed scale", () => {
    expect(MAPPING.mn9).toMatchObject({
      joints: ["proboscis"], kind: "extension", fullScaleHz: 60, maxDeg: 60, validation: "validated",
    });
    expect(MAPPING.mn9.note).toContain("Shiu");
  });
  it("abn1 drives both antennae as an illustrative 6 Hz oscillation", () => {
    expect(MAPPING.abn1).toMatchObject({
      joints: ["antenna_l", "antenna_r"], kind: "oscillation", oscillationHz: 6, fullScaleHz: 30, maxDeg: 25,
      validation: "illustrative",
    });
  });
  it("dnp01 raises both wings, exploratory", () => {
    expect(MAPPING.dnp01).toMatchObject({
      joints: ["wing_l", "wing_r"], kind: "extension", fullScaleHz: 100, maxDeg: 70, validation: "exploratory",
    });
  });
  it("has exactly the three first-version rows", () => {
    expect(Object.keys(MAPPING).sort()).toEqual(["abn1", "dnp01", "mn9"]);
  });
});

describe("poseFor extension", () => {
  it("scales linearly with a fixed full scale", () => {
    expect(poseFor({ mn9: 30 }, 0).proboscis).toBeCloseTo(30, 9);
    expect(poseFor({ mn9: 60 }, 0).proboscis).toBeCloseTo(60, 9);
    expect(poseFor({ dnp01: 50 }, 123)).toEqual({ wing_l: 35, wing_r: 35 });
  });
  it("clamps above full scale and at zero", () => {
    expect(poseFor({ mn9: 600 }, 0).proboscis).toBe(60);
    expect(poseFor({ mn9: 0 }, 0).proboscis).toBe(0);
  });
  it("a low dose looks weaker than a high dose (never normalized per run)", () => {
    expect(poseFor({ mn9: 4 }, 0).proboscis!).toBeLessThan(poseFor({ mn9: 63.3 }, 0).proboscis!);
    expect(poseFor({ mn9: 4 }, 0).proboscis).toBeCloseTo(4, 9);
  });
});

describe("poseFor oscillation", () => {
  it("peaks at intensity * maxDeg a quarter period in and crosses zero at half period", () => {
    const quarter = 1000 / 6 / 4;
    expect(poseFor({ abn1: 30 }, quarter).antenna_l).toBeCloseTo(25, 6);
    expect(poseFor({ abn1: 15 }, quarter).antenna_r).toBeCloseTo(12.5, 6);
    expect(poseFor({ abn1: 30 }, 0).antenna_l).toBeCloseTo(0, 9);
    expect(poseFor({ abn1: 30 }, 1000 / 12).antenna_l).toBeCloseTo(0, 6);
    expect(poseFor({ abn1: 30 }, (3 * 1000) / 24).antenna_l).toBeCloseTo(-25, 6);
  });
  it("repeats with a 1/6 s period", () => {
    const t = 37;
    expect(poseFor({ abn1: 20 }, t).antenna_l).toBeCloseTo(poseFor({ abn1: 20 }, t + 1000 / 6).antenna_l!, 6);
  });
  it("both antennae share the phase", () => {
    const p = poseFor({ abn1: 20 }, 40);
    expect(p.antenna_l).toBe(p.antenna_r);
  });
  it("clamps the amplitude at full scale", () => {
    expect(poseFor({ abn1: 3000 }, 1000 / 24).antenna_l).toBeCloseTo(25, 6);
  });
});

describe("poseFor invalid input", () => {
  it("non-finite and negative rates give 0", () => {
    for (const hz of [NaN, Infinity, -Infinity, -5]) {
      expect(poseFor({ mn9: hz }, 10).proboscis).toBe(0);
    }
  });
  it("a non-finite time gives 0 for oscillations", () => {
    expect(poseFor({ abn1: 30 }, NaN).antenna_l).toBe(0);
  });
  it("ignores unknown readouts without throwing and adds no keys", () => {
    expect(poseFor({ nope: 50, mn9: 30 }, 0)).toEqual({ proboscis: 30 });
    expect(poseFor({ nope: 50 }, 0)).toEqual({});
  });
  it("missing readouts give no key", () => {
    expect(poseFor({}, 0)).toEqual({});
    expect(poseFor({ mn9: 10 }, 0)).not.toHaveProperty("wing_l");
  });
  it("ignores inherited property names", () => {
    expect(poseFor({ toString: 5, constructor: 5 } as unknown as Record<string, number>, 0)).toEqual({});
  });
});

describe("poseFor combination and determinism", () => {
  it("is deterministic in its inputs", () => {
    const r = { mn9: 33, abn1: 12, dnp01: 80 };
    expect(poseFor(r, 77.5)).toEqual(poseFor({ ...r }, 77.5));
  });
  it("independent readouts drive their own joints", () => {
    const p = poseFor({ mn9: 60, dnp01: 100, abn1: 0 }, 0);
    expect(p).toMatchObject({ proboscis: 60, wing_l: 70, wing_r: 70, antenna_l: 0, antenna_r: 0 });
  });
});

describe("captionFor", () => {
  const SENTENCE = "mapping chosen by us; intensity follows the real readout";
  it("validated only: names the proboscis readout and still says the motion is ours", () => {
    const c = captionFor(["mn9"]);
    expect(c).toMatch(/proboscis extension/i);
    expect(c).toMatch(/proboscis motor neuron/);
    expect(c).toMatch(/validated/);
    expect(c).toMatch(/illustration/);
    expect(c.includes("\n")).toBe(false);
  });
  it("illustrative and exploratory sets carry their words and the sentence", () => {
    const a = captionFor(["abn1"]);
    expect(a).toMatch(/illustrative/);
    expect(a).toContain(SENTENCE);
    const d = captionFor(["dnp01"]);
    expect(d).toMatch(/exploratory/);
    expect(d).toContain(SENTENCE);
  });
  it("mixed sets name every non-validated level", () => {
    const m = captionFor(["mn9", "abn1", "dnp01"]);
    expect(m).toMatch(/illustrative/);
    expect(m).toMatch(/exploratory/);
    expect(m).toContain(SENTENCE);
    const mi = captionFor(["mn9", "abn1"]);
    expect(mi).toMatch(/illustrative/);
    expect(mi).not.toMatch(/exploratory/);
  });
  it("ignores unknown names and an empty set is an honest one-liner", () => {
    expect(captionFor(["zzz"])).toBe(captionFor([]));
    expect(captionFor([])).toContain(SENTENCE);
  });
  it("never uses forbidden words in any variant", () => {
    const names = ["mn9", "abn1", "dnp01", "zzz"];
    for (let mask = 0; mask < 16; mask++) {
      const c = captionFor(names.filter((_, i) => mask & (1 << i)));
      expect(c).not.toMatch(FORBIDDEN);
    }
    for (const row of Object.values(MAPPING)) expect(row.note).not.toMatch(FORBIDDEN);
  });
});

describe("sampleReadout", () => {
  it("empty series is 0", () => {
    expect(sampleReadout([], 10, 5)).toBe(0);
  });
  it("one bin is constant", () => {
    expect(sampleReadout([7], 10, -100)).toBe(7);
    expect(sampleReadout([7], 10, 100)).toBe(7);
  });
  it("interpolates linearly between bin centres", () => {
    // bin i covers [i*binMs, (i+1)*binMs); its centre is (i+0.5)*binMs
    expect(sampleReadout([0, 10], 10, 10)).toBeCloseTo(5, 9);
    expect(sampleReadout([0, 10], 10, 7.5)).toBeCloseTo(2.5, 9);
    expect(sampleReadout([0, 10, 20], 10, 15)).toBeCloseTo(10, 9);
  });
  it("clamps to the series ends", () => {
    expect(sampleReadout([3, 9], 10, -50)).toBe(3);
    expect(sampleReadout([3, 9], 10, 1e6)).toBe(9);
  });
  it("smooths with a centred moving average, clamped at the ends", () => {
    const s = [0, 0, 30, 0, 0];
    expect(sampleReadout(s, 10, 25, 3)).toBeCloseTo(10, 9);
    expect(sampleReadout(s, 10, 25, 1)).toBeCloseTo(30, 9);
    // edge values are replicated: samples at -5, 5, 15 ms -> (30 + 30 + 0) / 3
    expect(sampleReadout([30, 0, 0], 10, 5, 3)).toBeCloseTo(20, 9);
  });
  it("is safe on bad numbers", () => {
    expect(sampleReadout([1, 2], 0, 5)).toBe(0);
    expect(sampleReadout([1, 2], 10, NaN)).toBe(0);
    expect(sampleReadout([NaN, 4], 10, 5)).toBe(0);
  });
});
