import { describe, expect, it } from "vitest";
import { EXPANDED_KEY, loadExpanded, saveExpanded, summarizeTools, type TraceTool } from "../src/toolTrace";

const done = (name: string, ms: number): TraceTool => ({ name, state: "done", ms, startedAt: 0 });
const failed = (name: string, ms: number): TraceTool => ({ name, state: "failed", ms, startedAt: 0 });
const running = (name: string, startedAt: number, progress?: string): TraceTool => ({ name, state: "running", ms: null, startedAt, progress });

function memory(initial: Record<string, string> = {}) {
  const m = new Map(Object.entries(initial));
  return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => void m.set(k, v), m };
}

describe("summarizeTools", () => {
  it("is empty without tools", () => {
    expect(summarizeTools([], 0)).toBe("");
  });
  it("counts finished tools and sums their time", () => {
    expect(summarizeTools([done("a", 4000), done("b", 8200), done("c", 2000)], 0)).toBe("Used 3 tools · 14.2 s");
    expect(summarizeTools([done("a", 1500)], 0)).toBe("Used 1 tool · 1.5 s");
  });
  it("names failures", () => {
    expect(summarizeTools([done("a", 4000), failed("b", 1000), done("c", 1000)], 0)).toBe("3 tools · 1 failed · 6.0 s");
  });
  it("shows the current tool and its elapsed whole seconds while running", () => {
    expect(summarizeTools([done("search", 500), running("stimulate", 1000)], 7900)).toBe("running stimulate · 6 s");
  });
  it("prefers the progress label of the running tool", () => {
    expect(summarizeTools([running("silencing_screen", 0, "screen · 3/8")], 12_400)).toBe("running screen · 3/8 · 12 s");
  });
  it("never shows a negative elapsed time", () => {
    expect(summarizeTools([running("x", 5000)], 1000)).toBe("running x · 0 s");
  });
  it("counts an ended (cut-off) tool without a time as finished", () => {
    expect(summarizeTools([done("a", 2000), { name: "b", state: "ended", ms: null, startedAt: 0 }], 0)).toBe("Used 2 tools · 2.0 s");
  });
  it("with several running tools shows the latest started", () => {
    expect(summarizeTools([running("a", 0), running("b", 3000)], 5000)).toBe("running b · 2 s");
  });
});

describe("expanded preference", () => {
  it("defaults to collapsed", () => {
    expect(loadExpanded(memory())).toBe(false);
  });
  it("round-trips", () => {
    const s = memory();
    saveExpanded(true, s);
    expect(s.m.get(EXPANDED_KEY)).toBe("1");
    expect(loadExpanded(s)).toBe(true);
    saveExpanded(false, s);
    expect(loadExpanded(s)).toBe(false);
  });
  it("ignores garbage values", () => {
    expect(loadExpanded(memory({ [EXPANDED_KEY]: "maybe" }))).toBe(false);
  });
  it("tolerates blocked storage", () => {
    const blocked = { getItem: () => { throw new Error("denied"); }, setItem: () => { throw new Error("denied"); } };
    expect(loadExpanded(blocked)).toBe(false);
    expect(() => saveExpanded(true, blocked)).not.toThrow();
    expect(loadExpanded(null)).toBe(false);
  });
});
