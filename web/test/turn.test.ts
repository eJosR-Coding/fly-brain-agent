import { describe, expect, it } from "vitest";
import { guardHandlers, IDLE, isBusy, turnStep, type Turn } from "../src/turn";
import type { ChatHandlers } from "../src/chat";

const idle: Turn = IDLE;

describe("turnStep: valid transitions", () => {
  it("start from idle opens a pending turn with a fresh id", () => {
    expect(turnStep(idle, { type: "start", id: 1 })).toEqual({ k: "pending", id: 1 });
  });
  it("the first event moves pending to streaming", () => {
    expect(turnStep({ k: "pending", id: 1 }, { type: "event" })).toEqual({ k: "streaming", id: 1 });
  });
  it("events keep a streaming turn streaming", () => {
    const s: Turn = { k: "streaming", id: 1 };
    expect(turnStep(s, { type: "event" })).toBe(s);
  });
  it("stop moves a running turn to stopping, same id", () => {
    expect(turnStep({ k: "pending", id: 2 }, { type: "stop" })).toEqual({ k: "stopping", id: 2 });
    expect(turnStep({ k: "streaming", id: 2 }, { type: "stop" })).toEqual({ k: "stopping", id: 2 });
  });
  it("done returns every busy state to idle", () => {
    for (const s of [{ k: "pending", id: 1 }, { k: "streaming", id: 1 }, { k: "stopping", id: 1 }] as Turn[]) {
      expect(turnStep(s, { type: "done", id: 1 })).toEqual(idle);
    }
  });
});

describe("turnStep: invalid transitions are rejected (state unchanged)", () => {
  it("start while a turn is running is rejected", () => {
    for (const s of [{ k: "pending", id: 1 }, { k: "streaming", id: 1 }, { k: "stopping", id: 1 }] as Turn[]) {
      expect(turnStep(s, { type: "start", id: 2 })).toBe(s);
    }
  });
  it("done is idempotent", () => {
    expect(turnStep(idle, { type: "done", id: 1 })).toBe(idle);
  });
  it("a done of an older turn does not end the current one", () => {
    const s: Turn = { k: "streaming", id: 2 };
    expect(turnStep(s, { type: "done", id: 1 })).toBe(s);
  });
  it("event and stop on idle do nothing; event on stopping does not resurrect it", () => {
    expect(turnStep(idle, { type: "event" })).toBe(idle);
    expect(turnStep(idle, { type: "stop" })).toBe(idle);
    const s: Turn = { k: "stopping", id: 1 };
    expect(turnStep(s, { type: "event" })).toBe(s);
    expect(turnStep(s, { type: "stop" })).toBe(s);
  });
});

describe("isBusy", () => {
  it("is false only when idle", () => {
    expect(isBusy(idle)).toBe(false);
    expect(isBusy({ k: "pending", id: 1 })).toBe(true);
    expect(isBusy({ k: "stopping", id: 1 })).toBe(true);
  });
});

describe("guardHandlers", () => {
  function make() {
    const calls: string[] = [];
    const h = {
      onMeta: () => calls.push("meta"),
      onToolCall: () => calls.push("tool"),
      onHighlight: () => calls.push("hl"),
      onActivity: () => calls.push("act"),
      onAnswer: () => calls.push("answer"),
      onAnswerDelta: () => calls.push("delta"),
      onError: () => calls.push("error"),
      onDone: () => calls.push("done"),
    } as unknown as ChatHandlers;
    return { calls, h };
  }
  it("passes events through while the turn is current", () => {
    const { calls, h } = make();
    let state: Turn = { k: "pending", id: 1 };
    const g = guardHandlers(h, 1, () => state, () => { state = turnStep(state, { type: "event" }); });
    g.onMeta("c");
    g.onAnswer("x");
    expect(calls).toEqual(["meta", "answer"]);
    expect(state.k).toBe("streaming");
  });
  it("drops events after done (idle) but never blocks onDone itself", () => {
    const { calls, h } = make();
    const g = guardHandlers(h, 1, () => idle, () => undefined);
    g.onToolCall("t", {}, "i");
    g.onAnswerDelta?.("x");
    expect(calls).toEqual([]);
    g.onDone();
    expect(calls).toEqual(["done"]);
  });
  it("drops events of an older turn once a newer one is current", () => {
    const { calls, h } = make();
    const g = guardHandlers(h, 1, () => ({ k: "pending", id: 2 }), () => undefined);
    g.onHighlight({ selection_id: "s", total: 1, indices: [0] });
    expect(calls).toEqual([]);
  });
  it("while stopping only onError and onDone get through", () => {
    const { calls, h } = make();
    const g = guardHandlers(h, 1, () => ({ k: "stopping", id: 1 }), () => undefined);
    g.onAnswerDelta?.("x");
    g.onActivity({} as never);
    g.onError("aborted");
    g.onDone();
    expect(calls).toEqual(["error", "done"]);
  });
});
