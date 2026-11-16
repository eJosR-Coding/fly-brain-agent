import { describe, expect, it } from "vitest";
import { afterNewChat, EMPTY, legendOf, modeStage, StageHolder, type Stage } from "../src/stage";

const wave = (text: string): Stage => ({ kind: "wave", text });
const path = (text: string): Stage => ({ kind: "path", text });
const message = (text: string): Stage => ({ kind: "message", text });

describe("legendOf", () => {
  it("maps every kind to its text and dot", () => {
    expect(legendOf(EMPTY)).toEqual({ text: "no selection", active: false });
    expect(legendOf({ kind: "selection", total: 1234 })).toEqual({ text: "selected · 1,234", active: true });
    expect(legendOf({ kind: "selection", total: 0 })).toEqual({ text: "no matches", active: false });
    expect(legendOf({ kind: "selection", total: 0, msg: "try X" })).toEqual({ text: "try X", active: false });
    expect(legendOf({ kind: "selection", total: null, msg: "partners", lit: true })).toEqual({ text: "partners", active: true });
    expect(legendOf(wave("w"))).toEqual({ text: "w", active: true });
    expect(legendOf(path("p"))).toEqual({ text: "p", active: true });
    expect(legendOf(message("m"))).toEqual({ text: "m", active: false });
  });
});

describe("modeStage (player mode change)", () => {
  it("a mode legend takes the stage as a wave", () => {
    expect(modeStage(wave("a"), "diff", "a")).toEqual(wave("diff"));
    expect(modeStage(EMPTY, "diff", "a")).toEqual(wave("diff"));
  });
  it("null repaints the wave's own legend only while a wave is on stage", () => {
    expect(modeStage(wave("diff"), null, "base")).toEqual(wave("base"));
  });
  it("null does NOT repaint over a highlight, a path, a message or nothing", () => {
    for (const s of [{ kind: "selection", total: 5 } as Stage, path("p"), message("m"), EMPTY]) {
      expect(modeStage(s, null, "base")).toBe(s);
    }
  });
  it("null with no base legend keeps the stage", () => {
    const s = wave("x");
    expect(modeStage(s, null, "")).toBe(s);
  });
});

describe("afterNewChat", () => {
  it("clears only a path summary", () => {
    expect(afterNewChat(path("p"))).toEqual(EMPTY);
  });
  it("leaves a playing recording, a selection and messages alone", () => {
    for (const s of [wave("w"), { kind: "selection", total: 3 } as Stage, message("m"), EMPTY]) {
      expect(afterNewChat(s)).toBe(s);
    }
  });
});

describe("StageHolder lease (a timer that restores the previous stage)", () => {
  it("paints while it owns the stage and restores the previous one on release", () => {
    const h = new StageHolder();
    h.set(wave("w"));
    const l = h.lease();
    expect(l.paint(message("simulating 1"))).toBe(true);
    expect(l.paint(message("simulating 2"))).toBe(true);
    expect(h.current).toEqual(message("simulating 2"));
    expect(l.release()).toBe(true);
    expect(h.current).toEqual(wave("w"));
  });
  it("stops painting once someone else took the stage and does not restore over it", () => {
    const h = new StageHolder();
    h.set(EMPTY);
    const l = h.lease();
    l.paint(message("simulating"));
    h.set(wave("chat recording"));
    expect(l.paint(message("simulating 2"))).toBe(false);
    expect(h.current).toEqual(wave("chat recording"));
    expect(l.release()).toBe(false);
    expect(h.current).toEqual(wave("chat recording"));
  });
  it("release without ever painting changes nothing", () => {
    const h = new StageHolder();
    h.set(path("p"));
    expect(h.lease().release()).toBe(false);
    expect(h.current).toEqual(path("p"));
  });
  it("a second release is a no-op", () => {
    const h = new StageHolder();
    const l = h.lease();
    l.paint(message("m"));
    expect(l.release()).toBe(true);
    expect(l.release()).toBe(false);
  });
});
