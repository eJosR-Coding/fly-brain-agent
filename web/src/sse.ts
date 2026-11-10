/** Incremental Server-Sent Events parser (EventSource cannot POST, so we parse fetch streams). */

export interface SSEEvent {
  event: string;
  data: unknown;
}

export class SSEParser {
  private buffer = "";
  private event = "";
  private data: string[] = [];

  /** Feed a decoded text chunk; returns every event completed by it. */
  push(text: string): SSEEvent[] {
    this.buffer += text;
    const out: SSEEvent[] = [];
    // A lone trailing CR may be half of a CRLF split across chunks: keep it for the next push.
    const hold = this.buffer.endsWith("\r") ? 1 : 0;
    const ready = this.buffer.slice(0, this.buffer.length - hold);
    const lines = ready.split(/\r\n|\n|\r/);
    this.buffer = lines.pop()! + this.buffer.slice(this.buffer.length - hold);
    for (const line of lines) {
      const ev = this.line(line);
      if (ev) out.push(ev);
    }
    return out;
  }

  /** End of stream: a held-back trailing CR was a real line terminator. An unterminated partial line is dropped (per spec). */
  end(): SSEEvent[] {
    if (!this.buffer.endsWith("\r")) return [];
    const lines = this.buffer.split(/\r\n|\n|\r/);
    this.buffer = "";
    lines.pop(); // the empty remainder after the final CR
    const out: SSEEvent[] = [];
    for (const line of lines) {
      const ev = this.line(line);
      if (ev) out.push(ev);
    }
    return out;
  }

  private line(line: string): SSEEvent | null {
    if (line === "") return this.dispatch();
    if (line.startsWith(":")) return null; // comment / ping
    const i = line.indexOf(":");
    const field = i === -1 ? line : line.slice(0, i);
    let value = i === -1 ? "" : line.slice(i + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") this.event = value;
    else if (field === "data") this.data.push(value);
    return null;
  }

  private dispatch(): SSEEvent | null {
    const name = this.event || "message";
    const raw = this.data.join("\n");
    const had = this.data.length > 0 || this.event !== "";
    this.event = "";
    this.data = [];
    if (!had) return null;
    if (raw === "") return { event: name, data: null };
    try {
      return { event: name, data: JSON.parse(raw) };
    } catch {
      // policy: never throw mid-stream; surface it as an error event the UI can show
      return { event: "error", data: { message: `malformed JSON in "${name}" event` } };
    }
  }
}

/** Async-iterate the events of a fetch body. */
export async function* parseSSE(stream: ReadableStream<Uint8Array>): AsyncGenerator<SSEEvent> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  const parser = new SSEParser();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      yield* parser.push(decoder.decode(value, { stream: true }));
    }
    yield* parser.push(decoder.decode());
    yield* parser.end();
  } finally {
    reader.releaseLock();
  }
}
