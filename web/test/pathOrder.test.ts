import { describe, expect, it } from "vitest";
import { PathOrder } from "../src/pathOrder";

describe("PathOrder", () => {
  it("the newest path request wins", () => {
    const o = new PathOrder();
    const a = o.begin()!;
    const b = o.begin()!;
    expect(o.isCurrent(a)).toBe(false);
    expect(o.isCurrent(b)).toBe(true);
  });
  it("cancel (a highlight, a recording, a clear, a new chat) drops a waiting path", () => {
    const o = new PathOrder();
    const t = o.begin()!;
    o.cancel();
    expect(o.isCurrent(t)).toBe(false);
  });
  it("a path whose highlight was superseded never starts, and does not cancel an older valid one", () => {
    const o = new PathOrder();
    const older = o.begin()!;
    o.highlightSettled(false);
    expect(o.begin()).toBeNull();
    expect(o.isCurrent(older)).toBe(true);
  });
  it("a path whose highlight was applied starts", () => {
    const o = new PathOrder();
    o.highlightSettled(true);
    expect(o.begin()).not.toBeNull();
  });
  it("the verdict is per highlight: a later applied highlight re-enables paths", () => {
    const o = new PathOrder();
    o.highlightSettled(false);
    o.highlightSettled(true);
    expect(o.begin()).not.toBeNull();
  });
  it("a token is never current after cancel even when begin ran before", () => {
    const o = new PathOrder();
    o.highlightSettled(true);
    const t = o.begin()!;
    o.cancel();
    o.cancel();
    expect(o.isCurrent(t)).toBe(false);
  });
});

describe("PathOrder verdict lifetime", () => {
  it("a superseded verdict is consumed by the path it belonged to (a later path with no highlight still draws)", () => {
    const o = new PathOrder();
    o.highlightSettled(false);
    expect(o.begin()).toBeNull();
    expect(o.begin()).not.toBeNull();
  });
});
