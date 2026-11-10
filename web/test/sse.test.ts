import { describe, expect, it } from "vitest";
import { SSEParser, parseSSE } from "../src/sse";

const ev = (event: string, data: unknown) => ({ event, data });

describe("SSEParser", () => {
  it("parses one complete event", () => {
    const p = new SSEParser();
    expect(p.push('event: meta\ndata: {"a":1}\n\n')).toEqual([ev("meta", { a: 1 })]);
  });

  it("handles an event split across chunks, even mid-line", () => {
    const p = new SSEParser();
    expect(p.push("event: ans")).toEqual([]);
    expect(p.push('wer\ndata: {"text":"h')).toEqual([]);
    expect(p.push('i"}\n')).toEqual([]);
    expect(p.push("\n")).toEqual([ev("answer", { text: "hi" })]);
  });

  it("handles several events in one chunk", () => {
    const p = new SSEParser();
    const out = p.push('event: a\ndata: 1\n\nevent: b\ndata: 2\n\n');
    expect(out).toEqual([ev("a", 1), ev("b", 2)]);
  });

  it("handles CRLF line endings, including a CR/LF pair split across chunks", () => {
    const p = new SSEParser();
    expect(p.push('event: a\r\ndata: {"x":1}\r')).toEqual([]);
    expect(p.push("\n\r\n")).toEqual([ev("a", { x: 1 })]);
  });

  it("ignores comment and ping lines", () => {
    const p = new SSEParser();
    expect(p.push(": ping\n\n")).toEqual([]);
    expect(p.push(': keep-alive\nevent: a\ndata: 1\n\n')).toEqual([ev("a", 1)]);
  });

  it("defaults the event name to message and joins multi-line data", () => {
    const p = new SSEParser();
    expect(p.push('data: [1,\ndata: 2]\n\n')).toEqual([ev("message", [1, 2])]);
  });

  it("turns malformed JSON into an error event instead of throwing", () => {
    const p = new SSEParser();
    const out = p.push("event: highlight\ndata: {oops\n\nevent: done\ndata: {}\n\n");
    expect(out[0].event).toBe("error");
    expect((out[0].data as { message: string }).message).toContain("highlight");
    expect(out[1]).toEqual(ev("done", {}));
  });

  it("does not emit a trailing event that never got its blank line", () => {
    const p = new SSEParser();
    expect(p.push('event: a\ndata: 1\n')).toEqual([]);
  });
});

describe("parseSSE", () => {
  it("decodes a byte stream, including a multibyte char split across chunks", async () => {
    const bytes = new TextEncoder().encode('event: answer\ndata: {"text":"ñandú"}\n\n');
    const cut = bytes.indexOf(0xc3) + 1; // split inside the two-byte "ñ"
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(bytes.slice(0, cut));
        c.enqueue(bytes.slice(cut));
        c.close();
      },
    });
    const got = [];
    for await (const e of parseSSE(stream)) got.push(e);
    expect(got).toEqual([ev("answer", { text: "ñandú" })]);
  });
});

describe("end of stream", () => {
  it("flushes a final blank line that is a lone CR held back at the end", () => {
    const p = new SSEParser();
    expect(p.push("event: a\ndata: 1\n\r")).toEqual([]); // CR might be half of a CRLF
    expect(p.end()).toEqual([ev("a", 1)]);
  });

  it("parseSSE delivers an event terminated by a lone CR at the very end", async () => {
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(new TextEncoder().encode("event: done\rdata: {}\r\r"));
        c.close();
      },
    });
    const got = [];
    for await (const e of parseSSE(stream)) got.push(e);
    expect(got).toEqual([ev("done", {})]);
  });
});
