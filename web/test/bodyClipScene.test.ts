// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const made = vi.hoisted(() => ({ scenes: [] as unknown[], cameras: [] as unknown[], controls: [] as unknown[] }));

vi.mock("three", async (orig) => {
  const real = await orig<typeof import("three")>();
  class FakeRenderer {
    domElement = document.createElement("canvas");
    dispose() {}
    setPixelRatio() {}
    setClearColor() {}
    setSize() {}
    render(scene: unknown, camera: unknown) { made.scenes.push(scene); made.cameras.push(camera); }
  }
  return { ...real, WebGLRenderer: FakeRenderer };
});
vi.mock("three/addons/controls/OrbitControls.js", () => ({
  OrbitControls: class {
    target = { x: 0, y: 0, z: 0, copy() {}, add(v: { x: number; y: number; z: number }) { this.x += v.x; this.y += v.y; this.z += v.z; } };
    autoRotate = false;
    constructor() { made.controls.push(this); }
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

describe("createBodyScene world (root transform, floor, camera follow)", () => {
  const rot90 = [0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]; // 90 deg about +Y
  const world = (matrix: number[], offset: number[]) => ({ matrix, offset }) as never;
  const floor = () => lastScene().children.find((c) => c.name === "floor") as THREE.LineSegments;
  const cam = () => made.cameras[made.cameras.length - 1] as THREE.PerspectiveCamera;
  beforeEach(() => {
    made.scenes.length = 0;
    made.cameras.length = 0;
    vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => setTimeout(() => cb(0), 0));
    vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
    vi.stubGlobal("cancelAnimationFrame", (id: number) => clearTimeout(id));
  });
  afterEach(() => { for (const s of open.splice(0)) s.dispose(); vi.unstubAllGlobals(); });

  it("has a hidden floor grid at the body's lowest point until a world is set", async () => {
    createBodyScene(document.createElement("div"), makeBody({ bounds: { min: [0, -0.4, 0], max: [1, 1, 1] } }), theme);
    await tick();
    expect(floor()).toBeDefined();
    expect(floor().visible).toBe(false);
    expect(floor().position.y).toBeCloseTo(-0.4, 6);
  });
  it("applies the root transform to every group: rest, rig joints and leg bodies (composed with their own motion)", async () => {
    const scene = createBodyScene(document.createElement("div"), makeBody(), theme);
    await tick();
    scene.setClipFrame(new Map([["lf_coxa", MOVED]]));
    scene.setWorld(world(rot90, [0, 0, 0]));
    const want = new THREE.Matrix4().fromArray(rot90).multiply(new THREE.Matrix4().fromArray(MOVED)).elements;
    expect(matrix("lf_coxa")).toEqual(Array.from(want));
    expect(matrix("lf_tibia")).toEqual(rot90); // leg at rest: just the world
    expect(matrix("antenna_l")).toEqual(rot90);
    expect(Array.from(groupNamed("rest")!.matrix.elements)).toEqual(rot90);
    // a later leg frame keeps the world
    scene.setClipFrame(new Map([["lf_tibia", MOVED]]));
    expect(matrix("lf_tibia")).toEqual(Array.from(want));
    expect(matrix("lf_coxa")).toEqual(rot90);
    // and so does a pose
    scene.setPose({ antenna_l: 30 });
    expect(matrix("antenna_l")).not.toEqual(rot90);
  });
  it("null world returns every group to its own motion and hides the floor", async () => {
    const scene = createBodyScene(document.createElement("div"), makeBody(), theme);
    await tick();
    scene.setClipFrame(new Map([["lf_coxa", MOVED]]));
    scene.setWorld(world(rot90, [0, 0, 0]));
    expect(floor().visible).toBe(true);
    scene.setWorld(null);
    expect(matrix("lf_coxa")).toEqual(MOVED);
    expect(matrix("lf_tibia")).toEqual(IDENTITY);
    expect(floor().visible).toBe(false);
  });
  it("keeps the floor fixed in the world: snapped to whole cells under the thorax", async () => {
    const scene = createBodyScene(document.createElement("div"), makeBody({ radius: 5 }), theme); // cell = size / 5 = 2
    await tick();
    scene.setWorld(world(IDENTITY, [5, 0, -3]));
    expect(floor().position.x).toBe(4);
    expect(floor().position.z).toBe(-4);
    scene.setWorld(world(IDENTITY, [5.5, 0, -3.5]));
    expect(floor().position.x).toBe(4);
    expect(floor().position.z).toBe(-4);
  });
  it("glides the camera after the thorax (camera and focus move together), then settles", async () => {
    const scene = createBodyScene(document.createElement("div"), makeBody(), theme);
    await tick();
    const before = cam().position.clone();
    scene.setWorld(world(IDENTITY, [2, 0, 4]));
    await new Promise((r) => setTimeout(r, 400));
    const moved = cam().position.clone().sub(before);
    expect(moved.x).toBeCloseTo(2, 3);
    expect(moved.z).toBeCloseTo(4, 3);
    const settled = cam().position.clone();
    await new Promise((r) => setTimeout(r, 60));
    expect(cam().position.distanceTo(settled)).toBe(0); // settled: it no longer drifts
  });
  it("snaps the camera at once under reduced motion", async () => {
    const scene = createBodyScene(document.createElement("div"), makeBody(), theme, { reducedMotion: true });
    await tick();
    const before = cam().position.clone();
    scene.setWorld(world(IDENTITY, [2, 0, 4]));
    await tick();
    expect(cam().position.clone().sub(before).z).toBeCloseTo(4, 6);
  });
  it("does not auto-orbit while following", async () => {
    const scene = createBodyScene(document.createElement("div"), makeBody(), theme);
    await tick();
    scene.setWorld(world(IDENTITY, [1, 0, 1]));
    await tick();
    const controls = made.controls[made.controls.length - 1] as { autoRotate: boolean };
    expect(controls.autoRotate).toBe(false);
  });
});
