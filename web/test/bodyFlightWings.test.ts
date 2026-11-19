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
    joints: [{ id: "wing_l", parts: ["l_pedicel"], pivot: [0, 0, 0], axis: [0, 1, 0] }],
    mesh: { positions: new Float32Array(12), indices: new Uint16Array([0, 1, 2, 0, 1, 3, 1, 2, 3, 0, 2, 3]) },
    ...over,
  } as never;
}

const tick = () => new Promise((r) => setTimeout(r, 5));
const lastScene = () => made.scenes[made.scenes.length - 1] as THREE.Scene;
const groupNamed = (name: string) => lastScene().children.find((c) => c.name === name) as THREE.Group | undefined;

describe("flight wing pose on the scene", () => {
  beforeEach(() => {
    made.scenes.length = 0;
    vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => setTimeout(() => cb(0), 0));
    vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
    vi.stubGlobal("cancelAnimationFrame", (id: number) => clearTimeout(id));
  });
  afterEach(() => { for (const s of open.splice(0)) s.dispose(); vi.unstubAllGlobals(); });
  it("the wing group matrix changes with each stroke angle", async () => {
    const sc = createBodyScene(document.createElement("div"), makeBody(), theme);
    await tick();
    const seen = new Set<string>();
    for (const st of [-40, 0, 40, 75]) {
      sc.setPose({ wing_l_stroke: st, wing_l_dev: 0, wing_l_rot: 10 } as never);
      await tick();
      seen.add(Array.from(groupNamed("wing_l")!.matrix.elements).map((v) => v.toFixed(3)).join());
    }
    expect(seen.size).toBe(4);
  });
});
