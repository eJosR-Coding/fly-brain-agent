import { describe, expect, it } from "vitest";
import { streamChat, type ChatHandlers } from "../src/chat";

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
