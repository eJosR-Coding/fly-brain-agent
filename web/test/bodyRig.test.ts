import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { applyMatrix, rotationMatrix, splitByJoint } from "../src/bodyRig";
import { parseBody, parseBodyMesh, type BodyMeta } from "../src/bodyData";

const dist = (a: ArrayLike<number>, b: ArrayLike<number>) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

describe("rotationMatrix and applyMatrix", () => {
  it("is the identity at 0 degrees", () => {
    const p = [0.3, -0.2, 0.9];
    const q = applyMatrix(p, rotationMatrix([0, 1, 0], 0, [1, 2, 3]));
    expect(dist(q, p)).toBeLessThan(1e-12);
  });
  it("rotates 90 degrees right-handed about +Z through the origin", () => {
    const q = applyMatrix([1, 0, 0], rotationMatrix([0, 0, 1], Math.PI / 2, [0, 0, 0]));
    expect(dist(q, [0, 1, 0])).toBeLessThan(1e-12);
  });
  it("keeps the pivot fixed, keeps points on the axis fixed and preserves the distance to the pivot", () => {
    const pivot = [0.5, 1, -2], axis = [0, 0.6, 0.8];
    const m = rotationMatrix(axis, 1.1, pivot);
    expect(dist(applyMatrix(pivot, m), pivot)).toBeLessThan(1e-12);
    const onAxis = [pivot[0] + 2 * axis[0], pivot[1] + 2 * axis[1], pivot[2] + 2 * axis[2]];
    expect(dist(applyMatrix(onAxis, m), onAxis)).toBeLessThan(1e-12);
    const p = [3, -1, 4];
    expect(dist(applyMatrix(p, m), pivot)).toBeCloseTo(dist(p, pivot), 10);
  });
  it("normalizes a non-unit axis", () => {
    const a = applyMatrix([1, 0, 0], rotationMatrix([0, 0, 5], Math.PI / 2, [0, 0, 0]));
    expect(dist(a, [0, 1, 0])).toBeLessThan(1e-12);
  });
  it("returns a column-major 4x4 (translation in elements 12..14)", () => {
    const m = rotationMatrix([0, 0, 1], Math.PI, [1, 0, 0]);
    expect(m).toHaveLength(16);
    expect(m[12]).toBeCloseTo(2, 12);
    expect(m[15]).toBe(1);
  });
});

describe("splitByJoint (synthetic)", () => {
  // 3 parts, 1 and 2 segments and 1 segment; vertices 0..5; triangles 1 + 2 + 1
  const parts = [
    { name: "a", group: "g", side: "c", start: 0, count: 1 },
    { name: "b", group: "g", side: "l", start: 1, count: 2 },
    { name: "c", group: "g", side: "r", start: 3, count: 1 },
  ];
  const meshParts = [
    { name: "a", start: 0, count: 1 },
    { name: "b", start: 1, count: 2 },
    { name: "c", start: 3, count: 1 },
  ];
  const positions = Float32Array.from({ length: 4 * 6 }, (_, i) => i);
  const indices = Uint16Array.from([0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5]);
  const joints = [{ id: "j", parts: ["b", "a"] }];

  it("concatenates a joint's line segments and triangles in part order, the rest in part order", () => {
    const s = splitByJoint({ parts, meshParts, joints, positions, indices });
    // joint parts in PART order (a then b), not the joint's listing order
    expect(Array.from(s.joints.j.lines)).toEqual(Array.from(positions.slice(0, 18)));
    expect(Array.from(s.joints.j.indices!)).toEqual([0, 1, 2, 1, 2, 3, 2, 3, 4]);
    expect(Array.from(s.rest.lines)).toEqual(Array.from(positions.slice(18, 24)));
    expect(Array.from(s.rest.indices!)).toEqual([3, 4, 5]);
  });
  it("keeps the index dtype and the shared vertex numbering (no remap)", () => {
    const s = splitByJoint({ parts, meshParts, joints, positions, indices });
    expect(s.joints.j.indices).toBeInstanceOf(Uint16Array);
    const s32 = splitByJoint({ parts, meshParts, joints, positions, indices: Uint32Array.from(indices) });
    expect(s32.rest.indices).toBeInstanceOf(Uint32Array);
  });
  it("gives indices null when there is no mesh, and ignores a part claimed twice", () => {
    const s = splitByJoint({ parts, meshParts: [], joints: [{ id: "j", parts: ["a"] }, { id: "k", parts: ["a", "c"] }], positions, indices: null });
    expect(s.joints.j.indices).toBeNull();
    expect(s.rest.indices).toBeNull();
    expect(s.joints.j.lines.length).toBe(6);
    expect(s.joints.k.lines.length).toBe(6); // only "c": "a" already belongs to j
  });
});

describe("the committed asset", () => {
  const m = JSON.parse(readFileSync("public/fly-body.json", "utf8")) as BodyMeta;
  const toBuf = (f: string) => { const r = readFileSync(f); return r.buffer.slice(r.byteOffset, r.byteOffset + r.byteLength) as ArrayBuffer; };
  const body = parseBody(toBuf("public/fly-body.bin"), m);
  body.mesh = parseBodyMesh(toBuf("public/fly-body-mesh.bin"), m);
  const split = splitByJoint({ parts: body.parts, meshParts: body.meshParts, joints: body.joints, positions: body.positions, indices: body.mesh.indices });

  it("covers every segment and every triangle exactly once across the joints and the rest", () => {
    let segs = split.rest.lines.length / 6, tris = split.rest.indices!.length / 3;
    for (const j of body.joints) { segs += split.joints[j.id].lines.length / 6; tris += split.joints[j.id].indices!.length / 3; }
    expect(segs).toBe(body.segmentCount);
    expect(tris).toBe(m.mesh!.triangle_count);
    // the multiset of triangles matches too
    const key = (a: ArrayLike<number>) => { const t: string[] = []; for (let i = 0; i < a.length; i += 3) t.push(`${a[i]},${a[i + 1]},${a[i + 2]}`); return t; };
    const all = [...key(split.rest.indices!), ...body.joints.flatMap((j) => key(split.joints[j.id].indices!))].sort();
    expect(all).toEqual(key(body.mesh.indices).sort());
  });
  it("mirrors the left and right wing tips in X when both rotate by the same angle", () => {
    const tip = (id: "wing_l" | "wing_r") => {
      const j = body.joints.find((x) => x.id === id)!;
      const lines = split.joints[id].lines;
      // farthest vertex from the pivot = the wing tip
      let best = -1, bi = 0;
      for (let i = 0; i < lines.length; i += 3) { const d = dist([lines[i], lines[i + 1], lines[i + 2]], j.pivot); if (d > best) { best = d; bi = i; } }
      return applyMatrix([lines[bi], lines[bi + 1], lines[bi + 2]], rotationMatrix(j.axis, (40 * Math.PI) / 180, j.pivot));
    };
    const l = tip("wing_l"), r = tip("wing_r");
    // the folded wings cross the midline a little, so the sign of X is not asserted, only the mirror
    expect(Math.abs(l[0])).toBeGreaterThan(0.05);
    // symmetry of the asset itself is approximate: allow 3% of the tip separation
    const span = 0.03 * Math.abs(l[0] - r[0]);
    expect(Math.abs(l[0] + r[0])).toBeLessThan(span);
    expect(Math.abs(l[1] - r[1])).toBeLessThan(span);
    expect(Math.abs(l[2] - r[2])).toBeLessThan(span);
  });
});
