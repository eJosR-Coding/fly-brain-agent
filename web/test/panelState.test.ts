import { describe, expect, it } from "vitest";
import { COLLAPSED_KEY, loadCollapsed, saveCollapsed } from "../src/panelState";

function memory(initial: Record<string, string> = {}) {
  const data = { ...initial };
  return { data, getItem: (k: string) => data[k] ?? null, setItem: (k: string, v: string) => void (data[k] = v) };
}
const throwing = {
  getItem: () => { throw new Error("blocked"); },
  setItem: () => { throw new Error("blocked"); },
};

describe("panelState", () => {
  it("is not collapsed by default", () => {
    expect(loadCollapsed(memory())).toBe(false);
  });
  it("round-trips through the store", () => {
    const s = memory();
    saveCollapsed(true, s);
    expect(s.data[COLLAPSED_KEY]).toBe("1");
    expect(loadCollapsed(s)).toBe(true);
    saveCollapsed(false, s);
    expect(loadCollapsed(s)).toBe(false);
  });
  it("treats a corrupted value as not collapsed", () => {
    expect(loadCollapsed(memory({ [COLLAPSED_KEY]: "banana" }))).toBe(false);
  });
  it("never throws when storage throws", () => {
    expect(loadCollapsed(throwing)).toBe(false);
    expect(() => saveCollapsed(true, throwing)).not.toThrow();
  });
  it("works with no store at all", () => {
    expect(loadCollapsed(null)).toBe(false);
    expect(() => saveCollapsed(true, null)).not.toThrow();
  });
});
