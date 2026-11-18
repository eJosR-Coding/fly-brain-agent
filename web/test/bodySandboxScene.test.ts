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
import { FOLLOW_SPAN_MM, cubeBounds, cubeFitDistance, spanFitDistance } from "../src/bodyCube";

const open: BodyScene[] = [];
const createBodyScene: typeof create = (...args) => { const s = create(...args); open.push(s); return s; };
const theme = { highlightHex: "#ffffff", orbitSpeed: 1, orbitResumeMs: 100 } as never;
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const tick = () => new Promise((r) => setTimeout(r, 5));
const lastScene = () => made.scenes[made.scenes.length - 1] as THREE.Scene;
const cam = () => made.cameras[made.cameras.length - 1] as THREE.PerspectiveCamera;
const ctl = () => made.controls[made.controls.length - 1] as { target: { x: number; y: number; z: number } };
const byName = (n: string) => lastScene().getObjectByName(n);
const body = () => ({
  positions: new Float32Array(12), segmentCount: 2, centroid: [0, 0, 0], radius: 1.4, bounds: { min: [0, -0.4, 0], max: [1, 1, 1] },
  parts: [{ name: "c_thorax", group: "g", side: "c", start: 0, count: 2 }], meshParts: [], mesh: null,
}) as never;
const dist = () => cam().position.distanceTo(new THREE.Vector3(ctl().target.x, ctl().target.y, ctl().target.z));

describe("createBodyScene sandbox (glass cube)", () => {
  beforeEach(() => {
    made.scenes.length = 0; made.cameras.length = 0; made.controls.length = 0;
    vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => setTimeout(() => cb(0), 0));
    vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
    vi.stubGlobal("cancelAnimationFrame", (id: number) => clearTimeout(id));
  });
  afterEach(() => { for (const s of open.splice(0)) s.dispose(); vi.unstubAllGlobals(); });

  it("builds nothing for the sandbox until it is used", async () => {
    createBodyScene(document.createElement("div"), body(), theme);
    await tick();
    expect(byName("sandbox")).toBeUndefined();
  });
  it("shows the cube and replaces the loose floor grid with the cube's own", async () => {
    const scene = createBodyScene(document.createElement("div"), body(), theme);
    await tick();
    scene.setWorld({ matrix: IDENTITY, offset: [0, 0, 0] } as never);
    expect(byName("floor")!.visible).toBe(true);
    scene.setSandbox(true);
    expect(byName("sandbox")!.visible).toBe(true);
    expect(byName("floor")!.visible).toBe(false);
    scene.setWorld({ matrix: IDENTITY, offset: [1, 0, 1] } as never); // still hidden while walking in the cube
    expect(byName("floor")!.visible).toBe(false);
    scene.setSandbox(false);
    expect(byName("sandbox")!.visible).toBe(false);
    expect(byName("floor")!.visible).toBe(true);
  });
  it("closes the camera in on the fly and lets it back out to the whole cube; far plane covers the cube", async () => {
    const scene = createBodyScene(document.createElement("div"), body(), theme);
    await tick();
    const before = dist();
    const farBefore = cam().far;
    scene.setSandbox(true);
    expect(dist()).toBeCloseTo(spanFitDistance(FOLLOW_SPAN_MM, 40, cam().aspect), 3);
    expect(cam().far).toBeGreaterThan(cubeFitDistance(cubeBounds(-0.4), 40, cam().aspect) * 2);
    scene.setSandbox(false);
    expect(dist()).toBeCloseTo(before, 3);
    expect(cam().far).toBe(farBefore);
  });
  it("turning it on twice does not move the camera again", async () => {
    const scene = createBodyScene(document.createElement("div"), body(), theme);
    await tick();
    scene.setSandbox(true);
    const p = cam().position.clone();
    scene.setSandbox(true);
    expect(cam().position.equals(p)).toBe(true);
  });
  it("draws the objects layer and reports invalid objects", async () => {
    const scene = createBodyScene(document.createElement("div"), body(), theme);
    await tick();
    const r = scene.setObjects([{ id: "a", kind: "sugar", pos_mm: [1, 1, 1], size_mm: 2 }, { id: "x", kind: "salt", pos_mm: [0, 1, 0], size_mm: 2 }]);
    expect(r.errors).toHaveLength(1);
    expect(byName("objects")!.children.map((c) => c.name)).toEqual(["object:a"]);
  });
  it("releases the sandbox with the scene", async () => {
    const geo = vi.spyOn(THREE.BufferGeometry.prototype, "dispose");
    const scene = createBodyScene(document.createElement("div"), body(), theme);
    await tick();
    scene.setSandbox(true);
    geo.mockClear();
    scene.dispose();
    expect(geo.mock.calls.length).toBeGreaterThanOrEqual(5);
    geo.mockRestore();
  });
});
