import { afterEach, describe, expect, it, vi } from "vitest";
import { activeReadoutNames, createBodyDriver } from "../src/bodyDriver";
import type { JointId } from "../src/bodyPose";
import type { NamedReadout } from "../src/activity";

type Pose = Partial<Record<JointId, number>>;

const ro = (name: string | null, hz: number[]): NamedReadout => ({ name, label: name ?? "?", neurons: 1, rows: [], hz });
const act = (readouts: NamedReadout[], binMs = 10) => ({ readouts, binMs });

function setup(smoothBins?: number) {
  const calls: Pose[] = [];
  const driver = createBodyDriver({ setPose: (p) => void calls.push(p), smoothBins });
  return { calls, driver, last: () => calls[calls.length - 1] };
}

afterEach(() => vi.restoreAllMocks());

describe("createBodyDriver", () => {
  it("is at rest before any activity", () => {
    const { driver, calls } = setup();
    driver.onTime(100);
    driver.onTime(null);
    expect(calls.every((p) => Object.keys(p).length === 0)).toBe(true);
    expect(calls.length).toBeLessThanOrEqual(1);
  });

  it("a sugar-like mn9 plateau gives the full proboscis angle and 0 before the rise", () => {
    const { driver, last } = setup();
    driver.setActivity(act([ro("mn9", [0, 0, 0, 60, 60, 60, 60, 60])]));
    driver.onTime(5); // centre of bin 0
    expect(last().proboscis).toBe(0);
    driver.onTime(55); // centre of bin 5, on the plateau
    expect(last().proboscis).toBeCloseTo(60, 9);
  });

  it("an abn1 oscillation follows time (a quarter period apart)", () => {
    const { driver, last } = setup();
    driver.setActivity(act([ro("abn1", new Array(100).fill(30))]));
    driver.onTime(0);
    const atZero = last().antenna_l;
    driver.onTime(1000 / 6 / 4); // quarter of the 6 Hz period
    expect(atZero).toBeCloseTo(0, 9);
    expect(last().antenna_l).toBeCloseTo(25, 6);
    expect(last().antenna_r).toBeCloseTo(25, 6);
  });

  it("dnp01 drives both wings", () => {
    const { driver, last } = setup();
    driver.setActivity(act([ro("dnp01", [100, 100, 100])]));
    driver.onTime(15);
    expect(last()).toEqual({ wing_l: 70, wing_r: 70 });
  });

  it("combines several readouts at once", () => {
    const { driver, last } = setup();
    driver.setActivity(act([ro("mn9", [60, 60]), ro("dnp01", [50, 50])]));
    driver.onTime(10);
    expect(last()).toEqual({ proboscis: 60, wing_l: 35, wing_r: 35 });
  });

  it("ignores unknown and null-named readouts", () => {
    const { driver, last } = setup();
    driver.setActivity(act([ro(null, [60, 60]), ro("nope", [60, 60]), ro("mn9", [60, 60])]));
    driver.onTime(10);
    expect(last()).toEqual({ proboscis: 60 });
    driver.setActivity(act([ro(null, [60, 60]), ro("nope", [60, 60])]));
    driver.onTime(10);
    expect(last()).toEqual({});
  });

  it("the last readout wins on a duplicate name", () => {
    const { driver, last } = setup();
    driver.setActivity(act([ro("mn9", [60, 60]), ro("mn9", [30, 30])]));
    driver.onTime(10);
    expect(last().proboscis).toBeCloseTo(30, 9);
  });

  it("does not resend an identical pose", () => {
    const { driver, calls } = setup();
    driver.setActivity(act([ro("mn9", [60, 60, 60])]));
    driver.onTime(10);
    driver.onTime(10);
    driver.onTime(12.5); // constant series -> same pose
    expect(calls).toHaveLength(1);
    driver.onTime(null);
    driver.onTime(null);
    expect(calls).toHaveLength(2);
    expect(calls[1]).toEqual({});
  });

  it("sends a pose that becomes empty after a non-empty one exactly once", () => {
    const { driver, calls } = setup();
    driver.setActivity(act([ro("mn9", [60, 60])]));
    driver.onTime(10);
    driver.onTime(Number.NaN);
    driver.onTime(Number.NaN);
    expect(calls).toEqual([{ proboscis: 60 }, {}]);
  });

  it("null time and activity without usable readouts give rest", () => {
    const { driver, calls } = setup();
    driver.setActivity({ binMs: 10 });
    driver.onTime(10);
    driver.setActivity(act([]));
    driver.onTime(10);
    driver.setActivity(act([ro("mn9", [])]));
    driver.onTime(10);
    expect(calls.every((p) => Object.keys(p).length === 0)).toBe(true);
    driver.setActivity(act([ro("mn9", [60, 60])]));
    driver.onTime(10);
    driver.onTime(null);
    expect(calls[calls.length - 1]).toEqual({});
  });

  it("setActivity(null) returns to rest at once, once", () => {
    const { driver, calls } = setup();
    driver.setActivity(act([ro("mn9", [60, 60])]));
    driver.onTime(10);
    driver.setActivity(null);
    driver.setActivity(null);
    expect(calls).toEqual([{ proboscis: 60 }, {}]);
    driver.onTime(10);
    expect(calls).toHaveLength(2);
  });

  it("setActivity(null) at rest sends nothing", () => {
    const { driver, calls } = setup();
    driver.setActivity(null);
    expect(calls).toHaveLength(0);
  });

  it("a new activity replaces the previous one", () => {
    const { driver, last } = setup();
    driver.setActivity(act([ro("mn9", [60, 60])]));
    driver.onTime(10);
    driver.setActivity(act([ro("dnp01", [100, 100])]));
    driver.onTime(10);
    expect(last()).toEqual({ wing_l: 70, wing_r: 70 });
    expect(last().proboscis).toBeUndefined();
  });

  it("samples a 10 ms recording mid-bin and at the ends", () => {
    const { driver, last } = setup();
    driver.setActivity(act([ro("mn9", [0, 60])], 10));
    driver.onTime(-50); // before the start: clamped to the first bin
    expect(last().proboscis).toBe(0);
    driver.onTime(10); // midway between the two bin centres (5 and 15)
    expect(last().proboscis).toBeCloseTo(30, 9);
    driver.onTime(15);
    expect(last().proboscis).toBeCloseTo(60, 9);
    driver.onTime(5000); // after the end: clamped to the last bin
    expect(last().proboscis).toBeCloseTo(60, 9);
  });

  it("passes smoothBins to the sampler", () => {
    const sharp = setup(1);
    const smooth = setup(3);
    for (const s of [sharp, smooth]) {
      s.driver.setActivity(act([ro("mn9", [0, 0, 60, 0, 0])]));
      s.driver.onTime(25); // centre of the spike
    }
    expect(sharp.last().proboscis).toBeCloseTo(60, 9);
    expect(smooth.last().proboscis).toBeCloseTo(20, 9);
  });

  it("treats NaN and Infinity time like null", () => {
    const { driver, calls } = setup();
    driver.setActivity(act([ro("mn9", [60, 60])]));
    driver.onTime(10);
    driver.onTime(Number.POSITIVE_INFINITY);
    expect(calls[calls.length - 1]).toEqual({});
    driver.onTime(10);
    driver.onTime(Number.NaN);
    expect(calls[calls.length - 1]).toEqual({});
    expect(calls).toHaveLength(4);
  });

  it("an invalid binMs gives rest, never throws", () => {
    const { driver, last } = setup();
    driver.setActivity(act([ro("mn9", [60, 60])], 0));
    expect(() => driver.onTime(10)).not.toThrow();
    expect(last() ?? {}).toEqual({});
  });

  it("dispose silences everything", () => {
    const { driver, calls } = setup();
    driver.setActivity(act([ro("mn9", [60, 60])]));
    driver.onTime(10);
    driver.dispose();
    driver.onTime(20);
    driver.onTime(null);
    driver.setActivity(null);
    driver.setActivity(act([ro("dnp01", [100])]));
    driver.dispose();
    expect(calls).toHaveLength(1);
  });

  it("dispose during a setPose call stops later sends, including re-entrant ones", () => {
    const calls: Pose[] = [];
    let driver: ReturnType<typeof createBodyDriver>;
    driver = createBodyDriver({
      setPose: (p) => {
        calls.push(p);
        driver.dispose();
        driver.onTime(99); // re-entrant after dispose
        driver.setActivity(null);
      },
    });
    driver.setActivity(act([ro("mn9", [60, 60])]));
    driver.onTime(10);
    driver.onTime(20);
    expect(calls).toHaveLength(1);
  });

  it("survives a throwing setPose and logs each distinct message once", () => {
    const err = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const seen: Pose[] = [];
    let fail: string | null = "boom";
    const driver = createBodyDriver({
      setPose: (p) => {
        if (fail) throw new Error(fail);
        seen.push(p);
      },
    });
    driver.setActivity(act([ro("mn9", [60, 60])]));
    expect(() => driver.onTime(10)).not.toThrow();
    expect(() => driver.onTime(10)).not.toThrow();
    expect(err).toHaveBeenCalledTimes(1);
    fail = "other";
    driver.onTime(10);
    expect(err).toHaveBeenCalledTimes(2);
    fail = null;
    driver.onTime(10); // the failed pose is retried, not silently dropped
    expect(seen).toEqual([{ proboscis: 60 }]);
    driver.onTime(10);
    expect(seen).toHaveLength(1);
  });
});

describe("activeReadoutNames", () => {
  it("lists the mapped names that have data, once each, in order", () => {
    const names = activeReadoutNames(act([
      ro("dnp01", [1]), ro("mn9", [1]), ro("dnp01", [2]), ro("nope", [1]), ro(null, [1]), ro("abn1", []),
    ]));
    expect(names).toEqual(["dnp01", "mn9"]);
  });
  it("is empty for null, no readouts or inherited property names", () => {
    expect(activeReadoutNames(null)).toEqual([]);
    expect(activeReadoutNames({ binMs: 10 })).toEqual([]);
    expect(activeReadoutNames(act([ro("toString", [1]), ro("constructor", [1])]))).toEqual([]);
  });
});
