import { describe, expect, it, vi } from "vitest";
import { commitScene } from "../src/sceneLoad";

function setup() {
  const log: string[] = [];
  let created = 0;
  const make = () => {
    created++;
    const id = created;
    return { id, dispose: vi.fn(() => void log.push(`dispose${id}`)) };
  };
  return { log, make, count: () => created };
}

describe("commitScene", () => {
  it("creates, wires, then publishes last", () => {
    const { log, make } = setup();
    const s = commitScene({
      create: () => (log.push("create"), make()),
      wire: () => void log.push("wire"),
      publish: () => void log.push("publish"),
    });
    expect(log).toEqual(["create", "wire", "publish"]);
    expect(s.dispose).not.toHaveBeenCalled();
  });

  it("a wiring failure disposes the scene, undoes the wiring in reverse and never publishes", () => {
    const { log, make } = setup();
    const publish = vi.fn();
    expect(() =>
      commitScene({
        create: make,
        wire: (_s, undo) => {
          undo(() => void log.push("undo-a"));
          undo(() => void log.push("undo-b"));
          throw new Error("boom");
        },
        publish,
      }),
    ).toThrow("boom");
    expect(log).toEqual(["undo-b", "undo-a", "dispose1"]);
    expect(publish).not.toHaveBeenCalled();
  });

  it("a failing undo does not stop the rollback or hide the original error", () => {
    const { log, make } = setup();
    expect(() =>
      commitScene({
        create: make,
        wire: (_s, undo) => {
          undo(() => void log.push("undo-ok"));
          undo(() => { throw new Error("undo broke"); });
          throw new Error("wire broke");
        },
        publish: () => undefined,
      }),
    ).toThrow("wire broke");
    expect(log).toEqual(["undo-ok", "dispose1"]);
  });

  it("a creation failure has nothing to dispose", () => {
    const publish = vi.fn();
    expect(() => commitScene({ create: () => { throw new Error("no webgl"); }, wire: () => undefined, publish })).toThrow("no webgl");
    expect(publish).not.toHaveBeenCalled();
  });

  it("a retry after a wiring failure builds exactly one live scene", () => {
    const { make, count } = setup();
    let published: { id: number } | null = null;
    let fail = true;
    const attempt = () =>
      commitScene({
        create: make,
        wire: () => { if (fail) throw new Error("boom"); },
        publish: (s) => { published = s; },
      });
    expect(attempt).toThrow();
    expect(published).toBeNull(); // brainScene stays null: preview and control are kept by the caller
    fail = false;
    const s = attempt();
    expect(count()).toBe(2);
    expect(published).toBe(s);
    expect(s.dispose).not.toHaveBeenCalled(); // only the first (failed) one was disposed
  });
});
