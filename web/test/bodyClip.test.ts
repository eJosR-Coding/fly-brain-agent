import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { clipMatrices, loadWalkClip, parseWalkClip, type WalkClip } from "../src/bodyClip";
import { applyMatrix } from "../src/bodyRig";

const dist = (a: ArrayLike<number>, b: ArrayLike<number>) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const Q = 10000;

/** A tiny synthetic clip: one body "lf_coxa" at the neutral pose (pos in um, fly frame) and arbitrary frames. */
function raw(over: Record<string, unknown> = {}, frames: number[][] = [[0, 0, 0, Q, 0, 0, 0], [0, 0, 0, Q, 0, 0, 0]]) {
  return {
    format: "fly-walk/1",
    quat_scale: Q,
    thorax_origin_um: [0, 0, 0],
    bodies: ["lf_coxa"],
    legs: ["lf", "lm", "lh", "rf", "rm", "rh"],
    sample_rate_hz: 2,
    n_frames: frames.length,
    neutral: { pos_um: [[0, 0, 0]], quat: [[Q, 0, 0, 0]] },
    frames: frames.map((f) => [f]),
    adhesion: frames.map(() => [0, 0, 0, 0, 0, 0]),
    contact: frames.map(() => [0, 0, 0, 0, 0, 0]),
    ...over,
  };
}
const quatZ = (deg: number) => [Math.round(Math.cos((deg * Math.PI) / 360) * Q), 0, 0, Math.round(Math.sin((deg * Math.PI) / 360) * Q)];

describe("parseWalkClip validation", () => {
  it("accepts a well-formed clip", () => {
    const c = parseWalkClip(raw());
    expect(c.bodies).toEqual(["lf_coxa"]);
    expect(c.nFrames).toBe(2);
    expect(c.sampleRateHz).toBe(2);
    expect(c.durationS).toBe(1);
  });
  it.each([
    ["not an object", null],
    ["wrong format", raw({ format: "other/2" })],
    ["no bodies", raw({ bodies: [] })],
    ["duplicate bodies", raw({ bodies: ["lf_coxa", "lf_coxa"] })],
    ["non-string body", raw({ bodies: [3] })],
    ["bad sample rate", raw({ sample_rate_hz: 0 })],
    ["non-finite sample rate", raw({ sample_rate_hz: Infinity })],
    ["n_frames disagrees with frames", raw({ n_frames: 3 })],
    ["bad quat scale", raw({ quat_scale: 0 })],
    ["bad thorax origin", raw({ thorax_origin_um: [0, 0] })],
    ["neutral of the wrong size", raw({ neutral: { pos_um: [], quat: [] } })],
    ["a frame with the wrong body count", raw({ frames: [[[0, 0, 0, Q, 0, 0, 0], [0, 0, 0, Q, 0, 0, 0]], [[0, 0, 0, Q, 0, 0, 0]]] })],
    ["a body with the wrong number of numbers", raw({ frames: [[[0, 0, 0, Q, 0, 0]], [[0, 0, 0, Q, 0, 0, 0]]] })],
    ["a non-finite number", raw({}, [[0, Number.NaN, 0, Q, 0, 0, 0], [0, 0, 0, Q, 0, 0, 0]])],
    ["a zero quaternion", raw({}, [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, Q, 0, 0, 0]])],
  ])("rejects %s", (_name, input) => {
    expect(() => parseWalkClip(input)).toThrow();
  });
  it("accepts a clip without the optional adhesion and contact flags", () => {
    const r = raw();
    delete (r as Record<string, unknown>).adhesion;
    delete (r as Record<string, unknown>).contact;
    expect(() => parseWalkClip(r)).not.toThrow();
  });
});

describe("clipMatrices (synthetic)", () => {
  it("is the identity when the frame equals the neutral pose, at any time", () => {
    const c = parseWalkClip(raw());
    for (const t of [0, 0.3, 1, 7.77, -2.5]) {
      const m = clipMatrices(c, t).get("lf_coxa")!;
      for (let i = 0; i < 16; i++) expect(m[i]).toBeCloseTo(IDENTITY[i], 9);
    }
  });
  it("maps fly axes to the viewer axes with um -> mm: fly x (anterior) is viewer Z", () => {
    const c = parseWalkClip(raw({}, [[0, 0, 0, Q, 0, 0, 0], [100, 0, 0, Q, 0, 0, 0]]));
    const m = clipMatrices(c, 0.5).get("lf_coxa")!; // frame 1 at t = 1 / rate = 0.5 s
    expect(dist(applyMatrix([0, 0, 0], m), [0, 0, 0.1])).toBeLessThan(1e-9);
    const y = parseWalkClip(raw({}, [[0, 0, 0, Q, 0, 0, 0], [0, 100, 0, Q, 0, 0, 0]]));
    expect(dist(applyMatrix([0, 0, 0], clipMatrices(y, 0.5).get("lf_coxa")!), [0.1, 0, 0])).toBeLessThan(1e-9); // fly y (left) -> viewer X
    const z = parseWalkClip(raw({}, [[0, 0, 0, Q, 0, 0, 0], [0, 0, 100, Q, 0, 0, 0]]));
    expect(dist(applyMatrix([0, 0, 0], clipMatrices(z, 0.5).get("lf_coxa")!), [0, 0.1, 0])).toBeLessThan(1e-9); // fly z (dorsal) -> viewer Y
  });
  it("rotates like the fly frame: +90 deg about fly z takes fly x to fly y, i.e. viewer Z to viewer X", () => {
    const c = parseWalkClip(raw({}, [[0, 0, 0, Q, 0, 0, 0], [0, 0, 0, ...quatZ(90)]]));
    const m = clipMatrices(c, 0.5).get("lf_coxa")!;
    expect(dist(applyMatrix([0, 0, 1], m), [1, 0, 0])).toBeLessThan(1e-3);
    expect(dist(applyMatrix([0, 1, 0], m), [0, 1, 0])).toBeLessThan(1e-3); // the rotation axis is untouched
  });
  it("applies D = M(t) * M(neutral)^-1: a body whose neutral is offset moves by its frame delta only", () => {
    const r = raw({ neutral: { pos_um: [[500, 0, 0]], quat: [quatZ(90)] } }, [[500, 0, 0, ...quatZ(90)], [500, 0, 0, ...quatZ(180)]]);
    const c = parseWalkClip(r);
    const m = clipMatrices(c, 0.5).get("lf_coxa")!;
    // the body origin stays where it was (rotation about its own origin): the neutral origin in viewer mm is (0, 0, 0.5)
    expect(dist(applyMatrix([0, 0, 0.5], m), [0, 0, 0.5])).toBeLessThan(1e-3);
    // and a point 1 mm away along fly x of the body frame turns 90 deg about it
    expect(dist(applyMatrix([0, 0, 1.5], m), [0, 0, 0.5].map((v, i) => v + [1, 0, 0][i]) )).toBeLessThan(1e-3);
  });
  it("rotates about the thorax origin of the mesh frame, which stays fixed", () => {
    const r = raw({ thorax_origin_um: [0, 0, 1000] }, [[0, 0, 0, Q, 0, 0, 0], [0, 0, 0, ...quatZ(90)]]);
    const m = clipMatrices(parseWalkClip(r), 0.5).get("lf_coxa")!;
    const thoraxThree = [0, 1, 0]; // fly (0, 0, 1000 um) -> viewer (y, z, x) mm
    expect(dist(applyMatrix(thoraxThree, m), thoraxThree)).toBeLessThan(1e-9);
    // a point 1 mm anterior of the thorax origin goes 1 mm to the fly's left of it
    expect(dist(applyMatrix([0, 1, 1], m), [1, 1, 0])).toBeLessThan(1e-3);
  });
  it("interpolates positions linearly and rotations by slerp, and loops from the last frame back to the first", () => {
    const frames = [[0, 0, 0, Q, 0, 0, 0], [0, 0, 0, ...quatZ(90)], [100, 0, 0, ...quatZ(90)]]; // 3 frames, 2 Hz: 1.5 s
    const c = parseWalkClip(raw({ sample_rate_hz: 2 }, frames));
    expect(c.durationS).toBeCloseTo(1.5, 12);
    const half = clipMatrices(c, 0.25).get("lf_coxa")!; // halfway frame 0 -> 1: 45 degrees
    const p = applyMatrix([0, 0, 1], half);
    expect(dist(p, [Math.sin(Math.PI / 4), 0, Math.cos(Math.PI / 4)])).toBeLessThan(1e-3);
    const seam = clipMatrices(c, 1.25).get("lf_coxa")!; // halfway frame 2 -> frame 0 (wrap): translation 50 um
    expect(seam[14]).toBeGreaterThan(0.0001); // still moving back from 0.1 mm, the viewer Z of fly x
    expect(clipMatrices(c, 1.5 + 0.25).get("lf_coxa")).toEqual(half);
  });
  it("wraps any time into the loop, negative and non-finite included", () => {
    const c = parseWalkClip(raw({}, [[0, 0, 0, Q, 0, 0, 0], [100, 0, 0, Q, 0, 0, 0]]));
    const at = (t: number) => clipMatrices(c, t).get("lf_coxa")!;
    for (let i = 0; i < 16; i++) {
      expect(at(0.2 + 5)[i]).toBeCloseTo(at(0.2)[i], 9);
      expect(at(-0.8)[i]).toBeCloseTo(at(0.2)[i], 9);
      expect(at(Number.NaN)[i]).toBeCloseTo(at(0)[i], 9);
    }
  });
  it("takes the short way round between quaternions in opposite hemispheres", () => {
    const a = quatZ(10), b = quatZ(10).map((v) => -v);
    const c = parseWalkClip(raw({ neutral: { pos_um: [[0, 0, 0]], quat: [[Q, 0, 0, 0]] } }, [[0, 0, 0, ...a], [0, 0, 0, ...b]]));
    const m = clipMatrices(c, 0.25).get("lf_coxa")!;
    const q10 = clipMatrices(parseWalkClip(raw({}, [[0, 0, 0, ...a], [0, 0, 0, ...a]])), 0).get("lf_coxa")!;
    for (let i = 0; i < 16; i++) expect(m[i]).toBeCloseTo(q10[i], 6); // same rotation, not a spin through 180
  });
  it("returns a fresh column-major 4x4 per body", () => {
    const c = parseWalkClip(raw());
    const a = clipMatrices(c, 0), b = clipMatrices(c, 0);
    expect(a.get("lf_coxa")).toHaveLength(16);
    expect(a.get("lf_coxa")).not.toBe(b.get("lf_coxa"));
    expect(a.get("nope")).toBeUndefined();
  });
});

describe("loadWalkClip", () => {
  it("fetches fly-walk.json under the base and parses it", async () => {
    const urls: string[] = [];
    const fetchFn = (async (u: string) => { urls.push(u); return { ok: true, status: 200, json: async () => raw() }; }) as unknown as typeof fetch;
    expect((await loadWalkClip(fetchFn, "/x/")).bodies).toEqual(["lf_coxa"]);
    expect(urls).toEqual(["/x/fly-walk.json"]);
  });
  it("rejects on an HTTP error", async () => {
    const fetchFn = (async () => ({ ok: false, status: 404 })) as unknown as typeof fetch;
    await expect(loadWalkClip(fetchFn)).rejects.toThrow(/404/);
  });
});

describe("the committed fly-walk.json", () => {
  const json = JSON.parse(readFileSync("public/fly-walk.json", "utf8"));
  const body = JSON.parse(readFileSync("public/fly-body.json", "utf8")) as { parts: Array<{ name: string }> };
  const clip: WalkClip = parseWalkClip(json);

  it("has the 48 leg bodies, all of them parts of the body asset", () => {
    expect(clip.bodies).toHaveLength(48);
    const parts = new Set(body.parts.map((p) => p.name));
    for (const b of clip.bodies) expect(parts.has(b), b).toBe(true);
  });
  it("is 200 frames at 200 Hz: exactly 12 stepping cycles of 12 Hz in one second", () => {
    expect(clip.nFrames).toBe(200);
    expect(clip.sampleRateHz).toBe(200);
    expect(clip.durationS).toBe(1);
    expect(json.cycle_hz * clip.durationS).toBe(json.cycles);
  });
  it("records its provenance", () => {
    expect(json.provenance.flygym).toBe("2.1.1");
    expect(json.provenance.mujoco).toMatch(/^3\./);
    expect(json.provenance.drive).toEqual([1, 1]);
    expect(json.provenance.generator).toBe("scripts/export_flygym_walk.py");
  });
  it("loops seamlessly: the step from the last frame to the first is no bigger than an ordinary step", () => {
    const sample = (t: number) => clipMatrices(clip, t);
    const tip = [0, 0, 0]; // the body origins move; compare where each body's neutral origin is carried
    const move = (a: Map<string, number[]>, b: Map<string, number[]>) => {
      let worst = 0;
      for (const n of clip.bodies) worst = Math.max(worst, dist(applyMatrix(tip, a.get(n)!), applyMatrix(tip, b.get(n)!)));
      return worst;
    };
    const dt = 1 / clip.sampleRateHz;
    const seam = move(sample(clip.durationS - dt), sample(clip.durationS)); // last frame -> wrap -> frame 0
    let ordinary = 0;
    for (let f = 0; f + 1 < clip.nFrames; f++) ordinary = Math.max(ordinary, move(sample(f * dt), sample((f + 1) * dt)));
    expect(seam).toBeLessThanOrEqual(ordinary);
    // and the clip is periodic to within the measured seam error recorded at export
    expect(json.provenance.loop_seam.max_position_um).toBeLessThan(20);
    expect(json.provenance.loop_seam.max_rotation_deg).toBeLessThan(1.5);
  });
  it("moves the legs away from the neutral pose while walking", () => {
    const m = clipMatrices(clip, 0);
    let moved = 0;
    for (const n of clip.bodies) if (dist(applyMatrix([0, 0, 0], m.get(n)!), [0, 0, 0]) > 0.05) moved++;
    expect(moved).toBeGreaterThan(20); // well over half the bodies are displaced by more than 50 um from the neutral pose
  });
  it("makes the left and right legs mirror images: a right leg half a cycle later is the mirror (x -> -x) of the left", () => {
    const S = (p: number[]) => [-p[0], p[1], p[2]];
    const halfCycle = 0.5 / json.cycle_hz;
    for (const leg of ["f", "m", "h"]) {
      let anti = 0, inPhase = 0;
      for (let k = 0; k < 20; k++) {
        const t = (k / 20) * clip.durationS;
        // the passive, compliant distal tarsi are left out: their contact dynamics are not mirror-symmetric (hind tarsus5: ~0.4 mm)
        for (const link of ["coxa", "trochanterfemur", "tibia", "tarsus1"]) {
          // a point 1 mm from the body origin in a fixed direction: its image under D tracks the leg's motion
          const probe = [0.3, -0.2, 0.5];
          const l = applyMatrix(probe, clipMatrices(clip, t).get(`l${leg}_${link}`)!);
          const mirrored = S(applyMatrix(S(probe), clipMatrices(clip, t + halfCycle).get(`r${leg}_${link}`)!));
          const sameTime = S(applyMatrix(S(probe), clipMatrices(clip, t).get(`r${leg}_${link}`)!));
          anti = Math.max(anti, dist(l, mirrored));
          inPhase = Math.max(inPhase, dist(l, sameTime));
        }
      }
      expect(anti, `leg ${leg} antiphase mirror`).toBeLessThan(0.1);
      expect(inPhase, `leg ${leg} in phase`).toBeGreaterThan(anti * 2);
    }
  });
  it("keeps every leg connected: a child's neutral origin moves the same under its own D and its parent's D", () => {
    const links = ["coxa", "trochanterfemur", "tibia", "tarsus1", "tarsus2", "tarsus3", "tarsus4", "tarsus5"];
    for (const t of [0, 0.013, 0.05, 0.31, 0.77]) {
      const m = clipMatrices(clip, t);
      for (const leg of ["lf", "lm", "lh", "rf", "rm", "rh"]) {
        for (let i = 1; i < links.length; i++) {
          const child = clip.bodies.indexOf(`${leg}_${links[i]}`);
          // D acts on mesh-frame points: the thorax-relative neutral origin plus the thorax origin
          const origin = Array.from(clip.neutralPos.subarray(child * 3, child * 3 + 3)).map((v, k) => v + clip.thoraxOrigin[k]);
          const viaParent = applyMatrix(origin, m.get(`${leg}_${links[i - 1]}`)!);
          const viaSelf = applyMatrix(origin, m.get(`${leg}_${links[i]}`)!);
          expect(dist(viaParent, viaSelf), `${leg} ${links[i]} at ${t}`).toBeLessThan(0.005); // 5 um: rounding of the export (um positions, 4-decimal quaternions) plus interpolation
        }
      }
    }
  });
});
