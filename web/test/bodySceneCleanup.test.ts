// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";

const made = vi.hoisted(() => ({ renderer: null as null | { dispose: ReturnType<typeof vi.fn>; domElement: HTMLCanvasElement } }));

vi.mock("three", async (orig) => {
  const real = await orig<typeof import("three")>();
  class FakeRenderer {
    domElement = document.createElement("canvas");
    dispose = vi.fn();
    constructor() { made.renderer = this; }
    setPixelRatio() {}
    setClearColor() {}
    setSize() {}
    render() {}
  }
  return { ...real, WebGLRenderer: FakeRenderer };
});
vi.mock("three/addons/controls/OrbitControls.js", () => ({
  OrbitControls: class { constructor() { throw new Error("controls exploded"); } },
}));

import * as THREE from "three";
import { createBodyScene } from "../src/body";

describe("createBodyScene cleanup", () => {
  beforeEach(() => { made.renderer = null; });

  it("releases the renderer and removes the canvas when the setup throws after creating them", () => {
    const host = document.createElement("div");
    const body = { positions: new Float32Array(6), segmentCount: 1, centroid: [0, 0, 0], radius: 1, mesh: null } as never;
    const theme = { highlightHex: "#ffffff", orbitSpeed: 1, orbitResumeMs: 100 } as never;
    expect(() => createBodyScene(host, body, theme)).toThrow("controls exploded");
    expect(made.renderer?.dispose).toHaveBeenCalledTimes(1);
    expect(host.querySelector("canvas")).toBeNull();
  });

  it("releases every per-joint geometry and the shared materials when the setup throws", () => {
    const geo = vi.spyOn(THREE.BufferGeometry.prototype, "dispose");
    const mat = vi.spyOn(THREE.Material.prototype, "dispose");
    const host = document.createElement("div");
    const body = {
      positions: new Float32Array(12), segmentCount: 2, centroid: [0, 0, 0], radius: 1,
      parts: [{ name: "a", group: "g", side: "c", start: 0, count: 1 }, { name: "b", group: "g", side: "c", start: 1, count: 1 }],
      meshParts: [{ name: "a", start: 0, count: 1 }, { name: "b", start: 1, count: 1 }],
      joints: [{ id: "wing_l", parts: ["a"], pivot: [0, 0, 0], axis: [0, 1, 0] }],
      mesh: { positions: new Float32Array(9), indices: new Uint16Array([0, 1, 2, 0, 2, 1]) },
    } as never;
    const theme = { highlightHex: "#ffffff", orbitSpeed: 1, orbitResumeMs: 100 } as never;
    expect(() => createBodyScene(host, body, theme)).toThrow("controls exploded");
    // joint lines + joint occluder + static lines + static occluder + the floor grid
    expect(geo).toHaveBeenCalledTimes(5);
    // one line material + one occluder material + the floor's
    expect(mat).toHaveBeenCalledTimes(3);
    geo.mockRestore();
    mat.mockRestore();
  });
});
