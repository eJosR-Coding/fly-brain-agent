import { describe, expect, it } from "vitest";
import { DEFAULT_LAYOUT, LAYOUT_KEY, loadLayout, parseLayout, saveLayout, swap, toggleInset } from "../src/layoutState";

function memory(initial: Record<string, string> = {}) {
  const data = { ...initial };
  return { data, getItem: (k: string) => data[k] ?? null, setItem: (k: string, v: string) => void (data[k] = v) };
}
const throwing = {
  getItem: () => { throw new Error("blocked"); },
  setItem: () => { throw new Error("blocked"); },
};

describe("layoutState transitions", () => {
  it("defaults to the body large and the inset visible", () => {
    expect(DEFAULT_LAYOUT).toEqual({ main: "body", insetHidden: false });
  });
  it("swap exchanges the main pane and keeps the inset visibility", () => {
    expect(swap({ main: "body", insetHidden: false })).toEqual({ main: "brain", insetHidden: false });
    expect(swap({ main: "brain", insetHidden: true })).toEqual({ main: "body", insetHidden: true });
  });
  it("swap twice is the identity", () => {
    const s = { main: "body" as const, insetHidden: true };
    expect(swap(swap(s))).toEqual(s);
  });
  it("toggleInset flips only the visibility", () => {
    expect(toggleInset({ main: "brain", insetHidden: false })).toEqual({ main: "brain", insetHidden: true });
    expect(toggleInset({ main: "brain", insetHidden: true })).toEqual({ main: "brain", insetHidden: false });
  });
  it("never mutates its input", () => {
    const s = Object.freeze({ main: "body" as const, insetHidden: false });
    expect(() => { swap(s); toggleInset(s); }).not.toThrow();
  });
});

describe("parseLayout", () => {
  it("accepts a valid object", () => {
    expect(parseLayout({ main: "brain", insetHidden: true })).toEqual({ main: "brain", insetHidden: true });
  });
  it("accepts a valid JSON string", () => {
    expect(parseLayout('{"main":"brain","insetHidden":false}')).toEqual({ main: "brain", insetHidden: false });
  });
  it("falls back to the default on anything malformed", () => {
    for (const bad of [null, undefined, 3, "x", "{", "[]", [], { main: "sky", insetHidden: false }, { main: "body", insetHidden: "yes" }, { main: 1 }, "null"]) {
      expect(parseLayout(bad)).toEqual(DEFAULT_LAYOUT);
    }
  });
  it("fills a missing field from the default (an older stored shape)", () => {
    expect(parseLayout({ main: "brain" })).toEqual({ main: "brain", insetHidden: false });
    expect(parseLayout({ insetHidden: true })).toEqual({ main: "body", insetHidden: true });
  });
  it("ignores unknown extra fields", () => {
    expect(parseLayout({ main: "brain", insetHidden: false, junk: 1 })).toEqual({ main: "brain", insetHidden: false });
  });
  it("returns a fresh object, never the shared default", () => {
    expect(parseLayout(null)).not.toBe(DEFAULT_LAYOUT);
  });
});

describe("loadLayout / saveLayout", () => {
  it("returns the default for an empty store", () => {
    expect(loadLayout(memory())).toEqual(DEFAULT_LAYOUT);
  });
  it("round-trips through the store under flybrain.layout", () => {
    const s = memory();
    saveLayout({ main: "brain", insetHidden: true }, s);
    expect(LAYOUT_KEY).toBe("flybrain.layout");
    expect(s.data[LAYOUT_KEY]).toBeDefined();
    expect(loadLayout(s)).toEqual({ main: "brain", insetHidden: true });
  });
  it("treats a corrupted stored value as the default", () => {
    expect(loadLayout(memory({ [LAYOUT_KEY]: "banana" }))).toEqual(DEFAULT_LAYOUT);
  });
  it("never throws when storage throws, on get or on set", () => {
    expect(loadLayout(throwing)).toEqual(DEFAULT_LAYOUT);
    expect(() => saveLayout({ main: "brain", insetHidden: false }, throwing)).not.toThrow();
  });
  it("works with no store at all", () => {
    expect(loadLayout(null)).toEqual(DEFAULT_LAYOUT);
    expect(() => saveLayout(DEFAULT_LAYOUT, null)).not.toThrow();
  });
});
