import { describe, expect, it } from "vitest";
import {
  formatThinking, formatToolCall, newChatEnabled, formatTokens, formatUsage, parseUsage, streamChat, toolStatusText,
  type ChatHandlers,
} from "../src/chat";

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
    onToolCall: (t, _a, id) => log.push(id ? `tool:${t}#${id}` : `tool:${t}`),
    onToolResult: (id, ok, ms) => log.push(`result:${id}:${ok ? "ok" : "fail"}:${ms}`),
    onAnswerDelta: (t) => log.push(`delta:${t}`),
    onHighlight: (p) => log.push(`hl:${p.total}`),
    onActivity: (a) => log.push(`activity:${a.simulationId}:${a.nBins}x${a.binMs}`),
    onUsage: (u) => log.push(`usage:${u.calls}:${u.inputTokens}:${u.outputTokens}:${u.totalTokens}`),
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


describe("streamChat model choice and live events", () => {
  it("posts the chosen model only when there is one", async () => {
    const bodies: Record<string, unknown>[] = [];
    const f = (async (_u: string, init: RequestInit) => {
      bodies.push(JSON.parse(init.body as string));
      return sseResponse(["event: done\ndata: {}\n\n"]);
    }) as unknown as typeof fetch;
    await streamChat("a", null, recorder().h, f, "/api", () => null, "groq:fast");
    await streamChat("b", null, recorder().h, f);
    expect(bodies[0]).toEqual({ message: "a", model: "groq:fast" });
    expect(bodies[1]).toEqual({ message: "b" });
  });

  it("dispatches tool ids, tool results and answer deltas", async () => {
    const f = (async () =>
      sseResponse([
        'event: tool_call\ndata: {"id":"c1","tool":"search","args":{}}\n\n',
        'event: tool_result\ndata: {"id":"c1","tool":"search","ok":true,"ms":1234}\n\n',
        'event: tool_result\ndata: {"id":"c2","tool":"x","ok":false,"ms":5}\n\n',
        'event: answer_delta\ndata: {"text":"Hel"}\n\nevent: answer_delta\ndata: {"text":"lo"}\n\n',
        'event: answer\ndata: {"text":"Hello"}\n\nevent: done\ndata: {}\n\n',
      ])) as unknown as typeof fetch;
    const { log, h } = recorder();
    await streamChat("x", null, h, f);
    expect(log).toEqual([
      "tool:search#c1", "result:c1:ok:1234", "result:c2:fail:5", "delta:Hel", "delta:lo", "answer:Hello", "done",
    ]);
  });

  it("ignores malformed tool_result and answer_delta payloads", async () => {
    const f = (async () =>
      sseResponse([
        'event: tool_result\ndata: {"id":3,"ok":"yes","ms":"x"}\n\nevent: answer_delta\ndata: {"text":5}\n\nevent: done\ndata: {}\n\n',
      ])) as unknown as typeof fetch;
    const { log, h } = recorder();
    await streamChat("x", null, h, f);
    expect(log).toEqual(["done"]);
  });
});

describe("toolStatusText", () => {
  it("counts whole seconds while running", () => {
    expect(toolStatusText({ state: "running" }, 0)).toBe("running · 0 s");
    expect(toolStatusText({ state: "running" }, 4900)).toBe("running · 4 s");
    expect(toolStatusText({ state: "running" }, -5)).toBe("running · 0 s");
  });
  it("shows one decimal when done or failed", () => {
    expect(toolStatusText({ state: "done", ms: 12800 }, 0)).toBe("done · 12.8 s");
    expect(toolStatusText({ state: "failed", ms: 120 }, 0)).toBe("failed · 0.1 s");
  });
});

describe("formatTokens", () => {
  it("shows small counts whole and larger ones as k / M with one decimal", () => {
    expect(formatTokens(0)).toBe("0");
    expect(formatTokens(850)).toBe("850");
    expect(formatTokens(999)).toBe("999");
    expect(formatTokens(1000)).toBe("1.0k");
    expect(formatTokens(15_900)).toBe("15.9k");
    expect(formatTokens(18_240)).toBe("18.2k");
    expect(formatTokens(999_949)).toBe("999.9k");
    expect(formatTokens(1_250_000)).toBe("1.3M");
  });
});

describe("formatUsage", () => {
  it("is the calls, the total and the in/out split", () => {
    expect(formatUsage({ calls: 4, inputTokens: 15_900, outputTokens: 2_300, totalTokens: 18_200 })).toBe(
      "4 calls \u00b7 18.2k tokens (15.9k in \u00b7 2.3k out)",
    );
  });
  it("says 1 call, not 1 calls", () => {
    expect(formatUsage({ calls: 1, inputTokens: 800, outputTokens: 50, totalTokens: 850 })).toBe(
      "1 call \u00b7 850 tokens (800 in \u00b7 50 out)",
    );
  });
  it("derives the total when only in and out are known", () => {
    expect(formatUsage({ calls: 2, inputTokens: 1000, outputTokens: 500, totalTokens: null })).toContain("1.5k tokens");
  });
  it("drops the split when it is unknown but keeps the total", () => {
    expect(formatUsage({ calls: 2, inputTokens: null, outputTokens: null, totalTokens: 2_000 })).toBe(
      "2 calls \u00b7 2.0k tokens",
    );
  });
  it("is 'usage unavailable' when the provider reported nothing, or there is no usage", () => {
    expect(formatUsage({ calls: 3, inputTokens: null, outputTokens: null, totalTokens: null })).toBe("usage unavailable");
    expect(formatUsage(null)).toBe("usage unavailable");
  });
});

describe("parseUsage", () => {
  it("reads the wire payload", () => {
    expect(parseUsage({ calls: 2, input_tokens: 10, output_tokens: 5, total_tokens: 15 })).toEqual({
      calls: 2, inputTokens: 10, outputTokens: 5, totalTokens: 15,
    });
  });
  it("keeps null as null (never 0) and ignores junk fields", () => {
    expect(parseUsage({ calls: 1, input_tokens: null, output_tokens: "x", total_tokens: -3 })).toEqual({
      calls: 1, inputTokens: null, outputTokens: null, totalTokens: null,
    });
  });
  it("rejects a payload without a sane call count", () => {
    expect(parseUsage({})).toBeNull();
    expect(parseUsage({ calls: -1 })).toBeNull();
    expect(parseUsage({ calls: 1.5 })).toBeNull();
  });
});

describe("streamChat usage", () => {
  it("dispatches the usage event before the answer", async () => {
    const { log, h } = recorder();
    const f = (async () =>
      sseResponse([
        'event: usage\ndata: {"calls":2,"input_tokens":10,"output_tokens":5,"total_tokens":15}\n\n',
        'event: answer\ndata: {"text":"hi"}\n\nevent: done\ndata: {}\n\n',
      ])) as unknown as typeof fetch;
    await streamChat("x", null, h, f);
    expect(log).toEqual(["usage:2:10:5:15", "answer:hi", "done"]);
  });
  it("tolerates a handler-less consumer and a malformed usage event", async () => {
    const { log, h } = recorder();
    const f = (async () =>
      sseResponse(['event: usage\ndata: {"calls":"x"}\n\nevent: answer\ndata: {"text":"hi"}\n\n'])) as unknown as typeof fetch;
    await streamChat("x", null, h, f);
    expect(log).toEqual(["answer:hi", "done"]);
    const { onUsage: _drop, ...bare } = h;
    await expect(streamChat("x", null, bare, f)).resolves.toBeUndefined();
  });
});

describe("newChatEnabled", () => {
  it("is off while a turn streams, whatever else is true", () => {
    expect(newChatEnabled({ streaming: true, conversationId: "c1", messages: 4 })).toBe(false);
  });
  it("is off on an empty, untouched chat (nothing to reset)", () => {
    expect(newChatEnabled({ streaming: false, conversationId: null, messages: 0 })).toBe(false);
  });
  it("is on once there is a thread or anything in the log", () => {
    expect(newChatEnabled({ streaming: false, conversationId: "c1", messages: 0 })).toBe(true);
    expect(newChatEnabled({ streaming: false, conversationId: null, messages: 2 })).toBe(true);
  });
});
