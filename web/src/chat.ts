import { parseSSE } from "./sse";

export interface HighlightPayload {
  selection_id: string;
  total: number;
  indices: number[] | null;
  fetch?: string;
}

export interface ChatHandlers {
  onMeta(conversationId: string): void;
  onToolCall(tool: string, args: unknown): void;
  onHighlight(p: HighlightPayload): void;
  onAnswer(text: string): void;
  onError(message: string): void;
  onDone(): void;
}

/** POST /api/chat and dispatch the SSE events. Never throws; onDone fires exactly once. */
export async function streamChat(
  message: string,
  conversationId: string | null,
  h: ChatHandlers,
  fetchFn: typeof fetch = fetch,
  base = "/api",
): Promise<void> {
  let finished = false;
  const finish = () => {
    if (!finished) {
      finished = true;
      h.onDone();
    }
  };
  try {
    const body: { message: string; conversation_id?: string } = { message };
    if (conversationId) body.conversation_id = conversationId;
    const res = await fetchFn(`${base}/chat`, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "text/event-stream" },
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) {
      h.onError(`chat request failed: HTTP ${res.status}`);
      return;
    }
    for await (const ev of parseSSE(res.body)) {
      const d = (ev.data ?? {}) as Record<string, unknown>;
      switch (ev.event) {
        case "meta": h.onMeta(String(d.conversation_id)); break;
        case "tool_call": h.onToolCall(String(d.tool), d.args); break;
        case "highlight": h.onHighlight(d as unknown as HighlightPayload); break;
        case "answer": h.onAnswer(String(d.text)); break;
        case "error": h.onError(String(d.message)); break;
        case "done": finish(); break;
      }
    }
  } catch (e) {
    h.onError((e as Error).message);
  } finally {
    finish();
  }
}

/** DOM wiring for the chat panel. */
export function mountChat(
  root: HTMLElement,
  onHighlight: (p: HighlightPayload) => Promise<void> | void,
): void {
  root.innerHTML =
    '<div id="log"></div><form id="form"><input id="input" autocomplete="off" placeholder="ask about the fly brain" />' +
    '<button id="send" type="submit">send</button></form>';
  const log = root.querySelector<HTMLElement>("#log")!;
  const form = root.querySelector<HTMLFormElement>("#form")!;
  const input = root.querySelector<HTMLInputElement>("#input")!;
  const send = root.querySelector<HTMLButtonElement>("#send")!;
  let conversationId: string | null = null;

  const add = (cls: string, text: string): HTMLElement => {
    const el = document.createElement("div");
    el.className = `msg ${cls}`;
    el.textContent = text;
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
    return el;
  };

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    input.disabled = send.disabled = true;
    add("user", text);
    void streamChat(text, conversationId, {
      onMeta: (id) => { conversationId = id; },
      onToolCall: (tool, args) => add("tool", `> ${tool}(${JSON.stringify(args)})`),
      onHighlight: (p) => {
        const el = add("hl", `lit ${p.total.toLocaleString("en-US")} neurons`);
        const sw = document.createElement("span");
        sw.className = "sw accent";
        el.prepend(sw);
        void Promise.resolve(onHighlight(p)).catch((err) => add("error", `highlight failed: ${(err as Error).message}`));
      },
      onAnswer: (t) => add("answer", t),
      onError: (m) => add("error", m),
      onDone: () => {
        input.disabled = send.disabled = false;
        input.focus();
      },
    });
  });
}
