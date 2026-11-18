// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const made = vi.hoisted(() => ({ scenes: [] as unknown[] }));

vi.mock("three", async (orig) => {
  const real = await orig<typeof import("three")>();
  class FakeRenderer {
    domElement = document.createElement("canvas");
    dispose() {}
    setPixelRatio() {}
    setClearColor() {}
    setSize() {}
    render(scene: unknown) { made.scenes.push(scene); }
  }
  return { ...real, WebGLRenderer: FakeRenderer };
});
vi.mock("three/addons/controls/OrbitControls.js", () => ({
  OrbitControls: class {
    target = { copy() {} };
    addEventListener() {}
    update() { return false; }
    dispose() {}
  },
}));

import * as THREE from "three";
import { createBodyScene as create, type BodyScene } from "../src/body";

const open: BodyScene[] = [];
const createBodyScene: typeof create = (...args) => { const s = create(...args); open.push(s); return s; };

const theme = { highlightHex: "#ffffff", orbitSpeed: 1, orbitResumeMs: 100 } as never;
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const MOVED = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.5, 0.25, 0.125, 1];

/** thorax, left front coxa, left front tibia, one antenna part (a rig joint): 1 segment and 1 triangle each */
function makeBody(over: Record<string, unknown> = {}) {
  const names = ["c_thorax", "lf_coxa", "lf_tibia", "l_pedicel"];
  return {
    positions: new Float32Array(names.length * 6), segmentCount: names.length, centroid: [0, 0, 0], radius: 1,
    bounds: { min: [0, 0, 0], max: [1, 1, 1] },
    parts: names.map((name, i) => ({ name, group: "g", side: "c", start: i, count: 1 })),
    meshParts: names.map((name, i) => ({ name, start: i, count: 1 })),
    joints: [{ id: "antenna_l", parts: ["l_pedicel"], pivot: [0, 0, 0], axis: [0, 1, 0] }],
    mesh: { positions: new Float32Array(12), indices: new Uint16Array([0, 1, 2, 0, 1, 3, 1, 2, 3, 0, 2, 3]) },
    ...over,
  } as never;
}

const tick = () => new Promise((r) => setTimeout(r, 5));
const lastScene = () => made.scenes[made.scenes.length - 1] as THREE.Scene;
const groupNamed = (name: string) => lastScene().children.find((c) => c.name === name) as THREE.Group | undefined;
const matrix = (name: string) => Array.from(groupNamed(name)!.matrix.elements);
const drawn = () => lastScene().children.flatMap((g) => g.children.filter((o) => o instanceof THREE.LineSegments) as THREE.LineSegments[]);

describe("createBodyScene clip frames", () => {
  beforeEach(() => {
    made.scenes.length = 0;
    vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => setTimeout(() => cb(0), 0));
    vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
    vi.stubGlobal("cancelAnimationFrame", (id: number) => clearTimeout(id));
  });
  afterEach(() => { for (const s of open.splice(0)) s.dispose(); vi.unstubAllGlobals(); });

  it("gives every leg body its own group at rest (identity) and keeps every segment drawn exactly once", async () => {
    createBodyScene(document.createElement("div"), makeBody(), theme);
    await tick();
    for (const n of ["lf_coxa", "lf_tibia"]) {
      expect(groupNamed(n), n).toBeDefined();
      expect(matrix(n)).toEqual(IDENTITY);
    }
    const segments = drawn().reduce((n, l) => n + l.geometry.getAttribute("position").count / 2, 0);
    expect(segments).toBe(4);
    const tris = lastScene().children.flatMap((g) => g.children.filter((o) => o instanceof THREE.Mesh) as THREE.Mesh[]).reduce((n, m) => n + m.geometry.getIndex()!.count / 3, 0);
    expect(tris).toBe(4);
  });
  it("does not make a leg group out of a part a rig joint already moves", async () => {
    const body = makeBody({ joints: [{ id: "wing_l", parts: ["lf_tibia"], pivot: [0, 0, 0], axis: [0, 1, 0] }] });
    const scene = createBodyScene(document.createElement("div"), body, theme);
    await tick();
    expect(groupNamed("lf_tibia")).toBeUndefined();
    expect(groupNamed("lf_coxa")).toBeDefined();
    expect(scene.jointIds()).toEqual(["wing_l"]);
    const segments = drawn().reduce((n, l) => n + l.geometry.getAttribute("position").count / 2, 0);
    expect(segments).toBe(4);
  });
  it("applies a clip frame to the named leg groups only, ignores unknown names, and null returns to rest", async () => {
    const scene = createBodyScene(document.createElement("div"), makeBody(), theme);
    await tick();
    scene.setClipFrame(new Map([["lf_coxa", MOVED], ["zz_tarsus9", MOVED]]));
    expect(matrix("lf_coxa")).toEqual(MOVED);
    expect(matrix("lf_tibia")).toEqual(IDENTITY);
    scene.setClipFrame(null);
    expect(matrix("lf_coxa")).toEqual(IDENTITY);
  });
  it("replaces the whole frame: a body missing from the next frame returns to rest", async () => {
    const scene = createBodyScene(document.createElement("div"), makeBody(), theme);
    await tick();
    scene.setClipFrame(new Map([["lf_coxa", MOVED]]));
    scene.setClipFrame(new Map([["lf_tibia", MOVED]]));
    expect(matrix("lf_coxa")).toEqual(IDENTITY);
    expect(matrix("lf_tibia")).toEqual(MOVED);
  });
  it("asks for a redraw when the frame changes", async () => {
    const scene = createBodyScene(document.createElement("div"), makeBody(), theme);
    await tick();
    const before = made.scenes.length;
    scene.setClipFrame(new Map([["lf_coxa", MOVED]]));
    await tick();
    expect(made.scenes.length).toBeGreaterThan(before);
  });
  it("ignores a malformed matrix instead of corrupting the group", async () => {
    const scene = createBodyScene(document.createElement("div"), makeBody(), theme);
    await tick();
    scene.setClipFrame(new Map([["lf_coxa", [1, 2, 3]], ["lf_tibia", MOVED.map((v, i) => (i === 3 ? Number.NaN : v))]]));
    expect(matrix("lf_coxa")).toEqual(IDENTITY);
    expect(matrix("lf_tibia")).toEqual(IDENTITY);
  });
  it("still moves the joints with setPose, independently of the clip", async () => {
    const scene = createBodyScene(document.createElement("div"), makeBody(), theme);
    await tick();
    scene.setPose({ antenna_l: 90 });
    scene.setClipFrame(new Map([["lf_coxa", MOVED]]));
    const antenna = groupNamed("antenna_l")!;
    expect(antenna.matrix.elements[0]).toBeCloseTo(0, 6); // rotated 90 deg about Y
    expect(matrix("lf_coxa")).toEqual(MOVED);
  });
});
