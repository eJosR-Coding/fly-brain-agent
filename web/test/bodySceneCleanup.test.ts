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
});
