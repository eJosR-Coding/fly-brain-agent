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

describe("mountBody setPose", () => {
  const fakeScene = () => ({ invalidate() {}, dispose: vi.fn(), setPose: vi.fn(), jointIds: () => [] });

  it("stores a pose set before the scene exists and applies the latest once on creation", async () => {
    const scene = fakeScene();
    const m = mountBody(document.createElement("div"), { theme, load: async () => body, create: () => scene });
    m.setPose({ wing_l: 10 });
    m.setPose({ proboscis: 30 });
    expect(scene.setPose).not.toHaveBeenCalled();
    await flush();
    expect(scene.setPose).toHaveBeenCalledTimes(1);
    expect(scene.setPose).toHaveBeenCalledWith({ proboscis: 30 });
  });
  it("applies nothing on creation when no pose was set", async () => {
    const scene = fakeScene();
    mountBody(document.createElement("div"), { theme, load: async () => body, create: () => scene });
    await flush();
    expect(scene.setPose).not.toHaveBeenCalled();
  });
  it("forwards later poses to the scene", async () => {
    const scene = fakeScene();
    const m = mountBody(document.createElement("div"), { theme, load: async () => body, create: () => scene });
    await flush();
    m.setPose({ wing_r: 5 });
    expect(scene.setPose).toHaveBeenCalledWith({ wing_r: 5 });
  });
  it("is a no-op after remove()", async () => {
    const scene = fakeScene();
    const m = mountBody(document.createElement("div"), { theme, load: async () => body, create: () => scene });
    await flush();
    m.remove();
    m.setPose({ wing_r: 5 });
    expect(scene.setPose).not.toHaveBeenCalled();
  });
  it("does not break when a throwing create is followed by setPose, and still cleans up", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const host = document.createElement("div");
    const m = mountBody(host, { theme, load: async () => body, create: () => { throw new Error("x"); } });
    m.setPose({ wing_l: 1 });
    await flush();
    expect(() => m.setPose({ wing_l: 2 })).not.toThrow();
    expect(host.querySelector(".body-stage")).toBeNull();
  });
});

describe("mountBody setCaption", () => {
  const fakeScene = () => ({ invalidate() {}, dispose: vi.fn(), setPose: vi.fn(), jointIds: () => [] });

  it("stores a caption set before the scene exists and shows it once the scene is created", async () => {
    const host = document.createElement("div");
    const m = mountBody(host, { theme, load: async () => body, create: fakeScene });
    m.setCaption("first");
    m.setCaption("latest");
    expect(host.querySelector(".body-caption")).toBeNull();
    await flush();
    expect(host.querySelector(".body-caption")?.textContent).toBe("latest");
  });
  it("starts with the honest default caption", async () => {
    const host = document.createElement("div");
    mountBody(host, { theme, load: async () => body, create: fakeScene });
    await flush();
    const text = host.querySelector(".body-caption")?.textContent ?? "";
    expect(text).toContain("mapping chosen by us");
    expect(text).not.toContain("not driven by the brain");
  });
  it("updates the caption after creation", async () => {
    const host = document.createElement("div");
    const m = mountBody(host, { theme, load: async () => body, create: fakeScene });
    await flush();
    m.setCaption("while running");
    expect(host.querySelector(".body-caption")?.textContent).toBe("while running");
  });
  it("is a no-op after remove()", async () => {
    const host = document.createElement("div");
    const m = mountBody(host, { theme, load: async () => body, create: fakeScene });
    await flush();
    const cap = host.querySelector(".body-caption") as HTMLElement;
    m.remove();
    m.setCaption("late");
    expect(cap.textContent).not.toBe("late");
  });
});

describe("mountBody setClipFrame", () => {
  const fakeScene = () => ({ invalidate() {}, dispose: vi.fn(), setPose: vi.fn(), setClipFrame: vi.fn(), setWorld: vi.fn(), jointIds: () => [] });
  const frame = new Map([["lf_coxa", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]]);

  it("stores a frame set before the scene exists and applies the latest once on creation", async () => {
    const scene = fakeScene();
    const m = mountBody(document.createElement("div"), { theme, load: async () => body, create: () => scene });
    m.setClipFrame(new Map());
    m.setClipFrame(frame);
    expect(scene.setClipFrame).not.toHaveBeenCalled();
    await flush();
    expect(scene.setClipFrame).toHaveBeenCalledTimes(1);
    expect(scene.setClipFrame).toHaveBeenCalledWith(frame);
  });
  it("applies nothing on creation when no frame was set, and forwards later frames and null (rest)", async () => {
    const scene = fakeScene();
    const m = mountBody(document.createElement("div"), { theme, load: async () => body, create: () => scene });
    await flush();
    expect(scene.setClipFrame).not.toHaveBeenCalled();
    m.setClipFrame(frame);
    m.setClipFrame(null);
    expect(scene.setClipFrame.mock.calls).toEqual([[frame], [null]]);
  });
  it("is a no-op after remove() and never throws when the scene failed to create", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const scene = fakeScene();
    const m = mountBody(document.createElement("div"), { theme, load: async () => body, create: () => scene });
    await flush();
    m.remove();
    m.setClipFrame(frame);
    expect(scene.setClipFrame).not.toHaveBeenCalled();
    const broken = mountBody(document.createElement("div"), { theme, load: async () => body, create: () => { throw new Error("x"); } });
    broken.setClipFrame(frame);
    await flush();
    expect(() => broken.setClipFrame(frame)).not.toThrow();
  });
});

describe("mountBody setWorld", () => {
  const fakeScene = () => ({ invalidate() {}, dispose: vi.fn(), setPose: vi.fn(), setClipFrame: vi.fn(), setWorld: vi.fn(), jointIds: () => [] });
  const w = { matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 2, 1], offset: [1, 0, 2] as [number, number, number] };

  it("stores the latest world set before the scene exists and applies it once on creation", async () => {
    const scene = fakeScene();
    const m = mountBody(document.createElement("div"), { theme, load: async () => body, create: () => scene });
    m.setWorld(null);
    m.setWorld(w);
    await flush();
    expect(scene.setWorld.mock.calls).toEqual([[w]]);
    m.setWorld(null);
    expect(scene.setWorld).toHaveBeenLastCalledWith(null);
  });
  it("applies nothing on creation when no world was set, and is a no-op after remove()", async () => {
    const scene = fakeScene();
    const m = mountBody(document.createElement("div"), { theme, load: async () => body, create: () => scene });
    await flush();
    expect(scene.setWorld).not.toHaveBeenCalled();
    m.remove();
    m.setWorld(w);
    expect(scene.setWorld).not.toHaveBeenCalled();
  });
});

describe("mountBody setArena", () => {
  const arena = { kind: "glass_cube" as const, interiorMm: 10, boundsMm: [-5, 5, -5, 5] as [number, number, number, number], note: "" };
  const fake = () => ({ invalidate() {}, dispose: vi.fn(), setSandbox: vi.fn(), setArena: vi.fn() });
  it("applies an arena set before the scene exists (before the sandbox), then passes later ones through", async () => {
    const scene = fake();
    const m = mountBody(document.createElement("div"), { theme, load: async () => body, create: () => scene as never });
    m.setArena(arena);
    m.setSandbox(true);
    await flush();
    expect(scene.setArena.mock.calls).toEqual([[arena]]);
    expect(scene.setArena.mock.invocationCallOrder[0]).toBeLessThan(scene.setSandbox.mock.invocationCallOrder[0]);
    m.setArena(null);
    expect(scene.setArena.mock.calls.at(-1)).toEqual([null]);
  });
});

describe("mountBody setSandbox / setObjects", () => {
  const fakeScene = () => ({
    invalidate() {}, dispose: vi.fn(), setSandbox: vi.fn(), setObjects: vi.fn(() => ({ errors: ["x"] })),
  });
  const list = [{ id: "a", kind: "sugar", pos_mm: [0, 1, 0], size_mm: 2 }];

  it("applies the sandbox and the objects set before the scene exists, once, on creation", async () => {
    const scene = fakeScene();
    const m = mountBody(document.createElement("div"), { theme, load: async () => body, create: () => scene as never });
    m.setSandbox(true);
    expect(m.setObjects(list)).toEqual({ errors: [] }); // nothing to validate against yet
    await flush();
    expect(scene.setSandbox.mock.calls).toEqual([[true]]);
    expect(scene.setObjects.mock.calls).toEqual([[list]]);
  });
  it("passes later changes straight through and returns the scene's validation errors", async () => {
    const scene = fakeScene();
    const m = mountBody(document.createElement("div"), { theme, load: async () => body, create: () => scene as never });
    await flush();
    expect(scene.setSandbox).not.toHaveBeenCalled(); // never asked for: nothing is applied
    expect(scene.setObjects).not.toHaveBeenCalled();
    m.setSandbox(true);
    m.setSandbox(false);
    expect(scene.setSandbox.mock.calls).toEqual([[true], [false]]);
    expect(m.setObjects(list)).toEqual({ errors: ["x"] });
  });
  it("is a no-op after remove()", async () => {
    const scene = fakeScene();
    const m = mountBody(document.createElement("div"), { theme, load: async () => body, create: () => scene as never });
    await flush();
    m.remove();
    m.setSandbox(true);
    expect(m.setObjects(list)).toEqual({ errors: [] });
    expect(scene.setSandbox).not.toHaveBeenCalled();
    expect(scene.setObjects).not.toHaveBeenCalled();
  });
});
