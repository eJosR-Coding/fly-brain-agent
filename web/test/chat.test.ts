import { describe, expect, it } from "vitest";
import { formatThinking, formatToolCall, streamChat, type ChatHandlers } from "../src/chat";

function sseResponse(chunks: string[], status = 200): Response {
  const enc = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      for (const ch of chunks) c.enqueue(enc.encode(ch));
      c.close();
    },
  });
  return new Response(body, { status, headers: { "content-type": "text/event-stream" } });
}

function recorder() {
  const log: string[] = [];
  const h: ChatHandlers = {
    onMeta: (id) => log.push(`meta:${id}`),
    onToolCall: (t) => log.push(`tool:${t}`),
    onHighlight: (p) => log.push(`hl:${p.total}`),
    onActivity: (a) => log.push(`activity:${a.simulationId}:${a.nBins}x${a.binMs}`),
    onAnswer: (t) => log.push(`answer:${t}`),
    onError: (m) => log.push(`error:${m}`),
    onDone: () => log.push("done"),
  };
  return { log, h };
}

describe("streamChat", () => {
  it("dispatches events in order and posts the conversation id", async () => {
    let sent: { message: string; conversation_id?: string } | null = null;
    const f = (async (_u: string, init: RequestInit) => {
      sent = JSON.parse(init.body as string);
      return sseResponse([
        'event: meta\ndata: {"conversation_id":"c1"}\n\nevent: tool_call\ndata: {"tool":"search_neurons","args":{}}\n\n',
        'event: highlight\ndata: {"selection_id":"s","total":3,"indices":[1,2,3]}\n\nevent: answer\ndata: {"text":"hi"}\n\nevent: done\ndata: {}\n\n',
      ]);
    }) as unknown as typeof fetch;
    const { log, h } = recorder();
    await streamChat("hello", "c0", h, f);
    expect(sent).toEqual({ message: "hello", conversation_id: "c0" });
    expect(log).toEqual(["meta:c1", "tool:search_neurons", "hl:3", "answer:hi", "done"]);
  });

  it("reports an http error and still calls onDone", async () => {
    const f = (async () => new Response("bad", { status: 422 })) as unknown as typeof fetch;
    const { log, h } = recorder();
    await streamChat("x", null, h, f);
    expect(log[0]).toMatch(/^error:.*422/);
    expect(log.at(-1)).toBe("done");
  });

  it("reports a network failure and still calls onDone", async () => {
    const f = (async () => {
      throw new TypeError("network down");
    }) as unknown as typeof fetch;
    const { log, h } = recorder();
    await streamChat("x", null, h, f);
    expect(log).toEqual(["error:network down", "done"]);
  });

  it("calls onDone once when the stream ends without a done event", async () => {
    const f = (async () => sseResponse(['event: answer\ndata: {"text":"a"}\n\n'])) as unknown as typeof fetch;
    const { log, h } = recorder();
    await streamChat("x", null, h, f);
    expect(log).toEqual(["answer:a", "done"]);
  });
});

describe("streamChat payload validation", () => {
  const run = async (chunks: string[], pointCount?: number) => {
    const f = (async () => sseResponse(chunks)) as unknown as typeof fetch;
    const { log, h } = recorder();
    await streamChat("x", "c0", h, f, "/api", pointCount === undefined ? undefined : () => pointCount);
    return log;
  };
  const hl = (d: unknown) => `event: highlight\ndata: ${JSON.stringify(d)}\n\n`;
  const tail = 'event: answer\ndata: {"text":"ok"}\n\nevent: done\ndata: {}\n\n';

  it.each([
    ["negative index", { selection_id: "s", total: 2, indices: [1, -1] }],
    ["fractional index", { selection_id: "s", total: 1, indices: [1.5] }],
    ["index out of range", { selection_id: "s", total: 1, indices: [10] }],
    ["non-array indices", { selection_id: "s", total: 1, indices: "1,2" }],
    ["null indices without fetch", { selection_id: "s", total: 1, indices: null }],
    ["non-numeric total", { selection_id: "s", total: "3", indices: [1] }],
  ])("skips a highlight with %s, shows an error line, keeps streaming", async (_n, payload) => {
    const log = await run([hl(payload), tail], 10);
    expect(log[0]).toMatch(/^error:.*highlight/);
    expect(log).toContain("answer:ok");
    expect(log.at(-1)).toBe("done");
    expect(log.some((l) => l.startsWith("hl:"))).toBe(false);
  });

  it("accepts inline indices in range and a fetch url", async () => {
    const log = await run([hl({ selection_id: "s", total: 2, indices: [0, 9] }), hl({ selection_id: "t", total: 99, indices: null, fetch: "/selections/t/indices" }), tail], 10);
    expect(log).toEqual(["hl:2", "hl:99", "answer:ok", "done"]);
  });

  it("meta without a conversation_id does not set 'undefined'", async () => {
    const log = await run(['event: meta\ndata: {}\n\n', 'event: meta\ndata: {"conversation_id":""}\n\n', tail]);
    expect(log.some((l) => l.startsWith("meta:"))).toBe(false);
  });
});


describe("formatToolCall", () => {
  it("renders keyword arguments like a call", () => {
    expect(formatToolCall("search_neurons", { nt: "dopamine", side: "left" })).toBe('search_neurons(nt="dopamine", side="left")');
  });
  it("renders numbers and no arguments", () => {
    expect(formatToolCall("top", { n: 5 })).toBe("top(n=5)");
    expect(formatToolCall("ping", {})).toBe("ping()");
    expect(formatToolCall("ping", null)).toBe("ping()");
  });
  it("falls back to JSON for non-object args", () => {
    expect(formatToolCall("f", [1, 2])).toBe("f([1,2])");
  });
});

describe("formatThinking", () => {
  it("says the agent is working and counts whole seconds", () => {
    expect(formatThinking(0)).toBe("agent is thinking · 0 s");
    expect(formatThinking(12.7)).toBe("agent is thinking · 12 s");
  });

  it("warns that a slow provider may be the cause after 20 s", () => {
    expect(formatThinking(25)).toBe("agent is thinking · 25 s · the model provider may be slow");
  });

  it("never shows negative or non-finite time", () => {
    expect(formatThinking(-3)).toBe("agent is thinking · 0 s");
    expect(formatThinking(Number.NaN)).toBe("agent is thinking · 0 s");
  });
});

describe("streamChat activity events", () => {
  it("dispatches a valid activity event after the highlight", async () => {
    const f = (async () =>
      sseResponse([
        'event: highlight\ndata: {"selection_id":"s","total":3,"indices":[1,2,3]}\n\n',
        'event: activity\ndata: {"simulation_id":"sim_1","n_bins":100,"bin_ms":10}\n\nevent: done\ndata: {}\n\n',
      ])) as unknown as typeof fetch;
    const { log, h } = recorder();
    await streamChat("x", null, h, f);
    expect(log).toEqual(["hl:3", "activity:sim_1:100x10", "done"]);
  });

  it("reports an invalid activity event instead of playing it", async () => {
    const f = (async () =>
      sseResponse(['event: activity\ndata: {"simulation_id":"sim_1","n_bins":0,"bin_ms":10}\n\nevent: done\ndata: {}\n\n'])) as unknown as typeof fetch;
    const { log, h } = recorder();
    await streamChat("x", null, h, f);
    expect(log[0]).toMatch(/^error:ignored an invalid activity event: n_bins/);
    expect(log.at(-1)).toBe("done");
  });
});
