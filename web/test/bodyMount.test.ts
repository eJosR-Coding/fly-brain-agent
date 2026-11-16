// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { mountBody } from "../src/bodyMount";
import type { Theme } from "../src/theme";

const theme = { highlightHex: "#ffffff" } as unknown as Theme;
const body = { positions: new Float32Array(6), segmentCount: 1, centroid: [0, 0, 0], radius: 1 } as never;
const flush = () => new Promise((r) => setTimeout(r, 0));

afterEach(() => vi.restoreAllMocks());

describe("mountBody", () => {
  it("adds the stage once the scene is created and removes it on remove()", async () => {
    const host = document.createElement("div");
    const dispose = vi.fn();
    const m = mountBody(host, { theme, load: async () => body, create: () => ({ invalidate() {}, dispose }) });
    await flush();
    expect(host.querySelector(".body-stage .body-caption")).not.toBeNull();
    m.remove();
    expect(dispose).toHaveBeenCalledTimes(1);
    expect(host.querySelector(".body-stage")).toBeNull();
  });

  it("drops the stage, warns once and stays safe to remove when the scene throws", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const host = document.createElement("div");
    const m = mountBody(host, {
      theme,
      load: async () => body,
      create: () => { throw new Error("controls exploded"); },
    });
    await flush();
    expect(host.querySelector(".body-stage")).toBeNull();
    expect(warn).toHaveBeenCalledTimes(1);
    expect(() => m.remove()).not.toThrow();
    expect(() => m.remove()).not.toThrow();
  });

  it("warns and mounts nothing when the asset fails to load", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const host = document.createElement("div");
    mountBody(host, { theme, load: async () => { throw new Error("404"); } });
    await flush();
    expect(host.children.length).toBe(0);
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it("does not create a scene when removed before the load finishes", async () => {
    const host = document.createElement("div");
    const create = vi.fn();
    const m = mountBody(host, { theme, load: async () => body, create });
    m.remove();
    await flush();
    expect(create).not.toHaveBeenCalled();
    expect(host.children.length).toBe(0);
  });
});
