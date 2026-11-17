import { describe, expect, it, vi } from "vitest";
import { createBodyAnimation, DEFAULT_BODY_CAPTION } from "../src/bodyAnimation";
import { createBodyDriver } from "../src/bodyDriver";
import { captionFor } from "../src/bodyPose";
import type { NamedReadout } from "../src/activity";

const ro = (name: string | null, hz: number[]): NamedReadout => ({ name, label: name ?? "?", neurons: 1, rows: [], hz });
// 8 bins of 10 ms: 80 ms long
const act = (readouts?: NamedReadout[]) => ({ readouts, binMs: 10, durationMs: 80 });
const PLATEAU = [60, 60, 60, 60, 60, 60, 60, 60];

function setup(reducedMotion = false) {
  const poses: Record<string, number>[] = [];
  const captions: string[] = [];
  const driver = createBodyDriver({ setPose: (p) => void poses.push({ ...p }) });
  const anim = createBodyAnimation({ driver, setCaption: (t) => void captions.push(t), defaultCaption: "DEFAULT", reducedMotion });
  return { poses, captions, anim, last: () => poses[poses.length - 1] };
}

describe("createBodyAnimation", () => {
  it("sets the driver activity and the caption together", () => {
    const { anim, captions, last } = setup();
    anim.setActivity(act([ro("mn9", PLATEAU)]));
    expect(captions).toEqual([captionFor(["mn9"])]);
    anim.onTime(20);
    expect(last().proboscis).toBeCloseTo(60);
  });

  it("a recording with only mn9 gets the validated-only caption", () => {
    const { anim, captions } = setup();
    anim.setActivity(act([ro("mn9", PLATEAU)]));
    expect(captions[0]).toContain("validated");
    expect(captions[0]).not.toContain("illustrative");
  });

  it("a recording with abn1 mentions illustrative", () => {
    const { anim, captions } = setup();
    anim.setActivity(act([ro("mn9", PLATEAU), ro("abn1", PLATEAU)]));
    expect(captions[0]).toContain("illustrative");
  });

  it("null restores the default caption and rest", () => {
    const { anim, captions, last } = setup();
    anim.setActivity(act([ro("mn9", PLATEAU)]));
    anim.onTime(20);
    anim.setActivity(null);
    expect(captions[captions.length - 1]).toBe("DEFAULT");
    expect(last()).toEqual({});
  });

  it("a recording without readouts, or with only unmapped or unnamed ones, keeps the default caption and rest", () => {
    for (const readouts of [undefined, [], [ro(null, PLATEAU)], [ro("zzz", PLATEAU)]]) {
      const { anim, captions, poses } = setup();
      anim.setActivity(act(readouts));
      anim.onTime(20);
      expect(captions[captions.length - 1]).toBe("DEFAULT");
      expect(poses.every((p) => Object.keys(p).length === 0)).toBe(true);
    }
  });

  it("a recording set before the player exists is posed when the clock arrives later (replay)", () => {
    const { anim, poses, last } = setup();
    anim.setActivity(act([ro("mn9", PLATEAU)]));
    expect(poses).toEqual([]); // nothing moves without a clock
    anim.onTime(30);
    expect(last().proboscis).toBeCloseTo(60);
  });

  it("returns to rest when the playhead hides (null)", () => {
    const { anim, last } = setup();
    anim.setActivity(act([ro("mn9", PLATEAU)]));
    anim.onTime(30);
    anim.onTime(null);
    expect(last()).toEqual({});
  });

  it("returns to rest when playback reaches the end of the recording", () => {
    const { anim, last } = setup();
    anim.setActivity(act([ro("mn9", PLATEAU)]));
    anim.onTime(79);
    expect(last().proboscis).toBeGreaterThan(0);
    anim.onTime(80);
    expect(last()).toEqual({});
  });

  it("reduced motion: shows the static final pose at once and again for any later clock value, never rest while loaded", () => {
    const { anim, poses, last } = setup(true);
    anim.setActivity(act([ro("mn9", [0, 0, 0, 0, 0, 0, 0, 30])]));
    expect(last().proboscis).toBeGreaterThan(0); // final bin sampled, no clock needed
    const n = poses.length;
    anim.onTime(null); // the player emits null on load under reduced motion
    anim.onTime(80);
    expect(last().proboscis).toBeGreaterThan(0);
    expect(poses.length).toBe(n); // identical pose is not re-sent
  });

  it("reduced motion with no recording is at rest", () => {
    const { anim, last } = setup(true);
    anim.setActivity(act([ro("mn9", PLATEAU)]));
    anim.setActivity(null);
    anim.onTime(null);
    expect(last()).toEqual({});
  });

  it("never starts a timer of its own", () => {
    const raf = vi.fn(() => 0);
    vi.stubGlobal("requestAnimationFrame", raf);
    const st = vi.spyOn(globalThis, "setInterval");
    const to = vi.spyOn(globalThis, "setTimeout");
    const { anim } = setup(true);
    anim.setActivity(act([ro("mn9", PLATEAU)]));
    anim.onTime(10);
    anim.dispose();
    expect(raf).not.toHaveBeenCalled();
    expect(st).not.toHaveBeenCalled();
    expect(to).not.toHaveBeenCalled();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("dispose rests the body, restores the caption and ignores later calls", () => {
    const { anim, captions, poses, last } = setup();
    anim.setActivity(act([ro("mn9", PLATEAU)]));
    anim.onTime(20);
    anim.dispose();
    expect(last()).toEqual({});
    expect(captions[captions.length - 1]).toBe("DEFAULT");
    const n = poses.length, c = captions.length;
    anim.setActivity(act([ro("mn9", PLATEAU)]));
    anim.onTime(20);
    expect(poses.length).toBe(n);
    expect(captions.length).toBe(c);
  });
});

describe("DEFAULT_BODY_CAPTION", () => {
  it("credits the mesh and says the mapping is ours", () => {
    expect(DEFAULT_BODY_CAPTION).toContain("NeuroMechFly v2");
    expect(DEFAULT_BODY_CAPTION).toContain("Apache-2.0");
    expect(DEFAULT_BODY_CAPTION).toContain("mapping chosen by us");
  });
  it("never talks about death, pain or feelings", () => {
    const banned = /\b(die|dies|dead|death|overdose|poison|kill|suffer|pain|feel|feels|feeling)\b/i;
    expect(DEFAULT_BODY_CAPTION).not.toMatch(banned);
    for (const names of [["mn9"], ["abn1"], ["dnp01"], ["mn9", "abn1", "dnp01"]]) expect(captionFor(names)).not.toMatch(banned);
  });
});
