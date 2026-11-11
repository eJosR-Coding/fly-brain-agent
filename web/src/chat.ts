import { parseActivityEvent, type ActivityEvent } from "./activity";
import { renderMarkdown } from "./markdown";
import { loadStoredModel, optionLabel, pickModel, saveStoredModel, type ModelList } from "./models";
import { parseSSE } from "./sse";

export interface HighlightPayload {
  selection_id: string;
  total: number;
  indices: number[] | null;
  fetch?: string;
}

export interface ChatHandlers {
  onMeta(conversationId: string): void;
  /** a tool started; `id` ties it to its later result (absent on older backends) */
  onToolCall(tool: string, args: unknown, id?: string): void;
  onToolResult?(id: string, ok: boolean, ms: number): void;
  /** a chunk of the final answer; `onAnswer` still delivers the full text */
  onAnswerDelta?(text: string): void;
  onHighlight(p: HighlightPayload): void;
  onActivity(a: ActivityEvent): void;
  onAnswer(text: string): void;
  onError(message: string): void;
  onDone(): void;
}

/** `tool(k="v", n=5)`: how a tool call is shown in the trace. */
/** Pending-state text while a turn runs; after 20 s hint that the provider may be slow. */
export function formatThinking(seconds: number): string {
  const s = Number.isFinite(seconds) && seconds > 0 ? Math.floor(seconds) : 0;
  const base = `agent is thinking \u00b7 ${s} s`;
  return s >= 20 ? `${base} \u00b7 the model provider may be slow` : base;
}

export type ToolState = { state: "running" } | { state: "done" | "failed"; ms: number };

/** Live text of a tool line: whole seconds while running, one decimal once finished. */
export function toolStatusText(t: ToolState, runningMs: number): string {
  if (t.state === "running") return `running \u00b7 ${Math.floor(Math.max(0, runningMs) / 1000)} s`;
  return `${t.state} \u00b7 ${(t.ms / 1000).toFixed(1)} s`;
}

export function formatToolCall(tool: string, args: unknown): string {
  if (typeof args === "object" && args !== null && !Array.isArray(args)) {
    const inner = Object.entries(args as Record<string, unknown>)
      .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
      .join(", ");
    return `${tool}(${inner})`;
  }
  return args === null || args === undefined ? `${tool}()` : `${tool}(${JSON.stringify(args)})`;
}

/** Check a highlight payload from the wire. Returns the typed payload, or why it is invalid. */
export function parseHighlight(d: Record<string, unknown>, pointCount: number | null): HighlightPayload | string {
  if (typeof d.total !== "number" || !Number.isFinite(d.total) || d.total < 0) return "total is not a number";
  const { indices } = d;
  if (indices === null || indices === undefined) {
    if (typeof d.fetch !== "string" || !d.fetch) return "no indices and no fetch url";
  } else if (!Array.isArray(indices)) {
    return "indices is not an array";
  } else {
    for (const i of indices) {
      if (typeof i !== "number" || !Number.isInteger(i) || i < 0) return `bad index ${JSON.stringify(i)}`;
      if (pointCount !== null && i >= pointCount) return `index ${i} is outside 0..${pointCount - 1}`;
    }
  }
  return {
    selection_id: String(d.selection_id ?? ""),
    total: d.total,
    indices: (indices as number[] | undefined) ?? null,
    fetch: typeof d.fetch === "string" ? d.fetch : undefined,
  };
}

/** POST /api/chat and dispatch the SSE events. Never throws; onDone fires exactly once. */
export async function streamChat(
  message: string,
  conversationId: string | null,
  h: ChatHandlers,
  fetchFn: typeof fetch = fetch,
  base = "/api",
  getPointCount: () => number | null = () => null,
  model: string | null = null,
): Promise<void> {
  let finished = false;
  const finish = () => {
    if (!finished) {
      finished = true;
      h.onDone();
    }
  };
  try {
    const body: { message: string; conversation_id?: string; model?: string } = { message };
    if (conversationId) body.conversation_id = conversationId;
    if (model) body.model = model;
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
        case "meta":
          if (typeof d.conversation_id === "string" && d.conversation_id) h.onMeta(d.conversation_id);
          break;
        case "tool_call": h.onToolCall(String(d.tool), d.args, typeof d.id === "string" ? d.id : undefined); break;
        case "tool_result":
          if (typeof d.id === "string" && typeof d.ok === "boolean" && typeof d.ms === "number") {
            h.onToolResult?.(d.id, d.ok, d.ms);
          }
          break;
        case "answer_delta":
          if (typeof d.text === "string" && d.text) h.onAnswerDelta?.(d.text);
          break;
        case "highlight": {
          const p = parseHighlight(d, getPointCount());
          if (typeof p === "string") h.onError(`ignored an invalid highlight: ${p}`);
          else h.onHighlight(p);
          break;
        }
        case "activity": {
          const a = parseActivityEvent(d);
          if (typeof a === "string") h.onError(`ignored an invalid activity event: ${a}`);
          else h.onActivity(a);
          break;
        }
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

export const HINT = "Try: How many serotonin neurons are there?";
const ARROW_UP = "M6.357 2.184q-.079 .012-.146 .06-.063 .044-2.012 1.993l-1.546 1.574q-.339 .337-.431 .467-.051 .079-.051 .184v.038q0 .222 .168 .39 .105 .105 .26 .137 .156 .032 .299-.019 .079-.026 .298-.241 .222-.216 1.276-1.267 1.469-1.469 1.482-1.469 .013 0 .013 3.183v2.796q0 .416 .032 .495 .034 .077 .111 .156 .079 .076 .174 .124 .099 .044 .216 .044 .117 0 .213-.044 .098-.048 .174-.124 .08-.079 .111-.156 .035-.079 .035-.495v-2.796q0-3.183 .013-3.183 .013 0 1.482 1.469 1.054 1.051 1.273 1.267 .222 .215 .301 .241 .143 .051 .299 .019 .155-.032 .26-.137 .168-.168 .168-.39v-.038q0-.105-.051-.184-.092-.13-.431-.479l-1.559-1.562q-1.952-1.961-2.015-1.993-.063-.035-.149-.06-.086-.026-.13-.026-.045 0-.137 .026z";

export interface ChatOptions {
  onHighlight: (p: HighlightPayload) => Promise<void> | void;
  /** a simulation finished during the turn: fetch and play its activity */
  onActivity?: (a: ActivityEvent) => Promise<void> | void;
  getPointCount?: () => number | null;
  /** true while an SSE turn is running */
  onStreaming?: (streaming: boolean) => void;
  /** the picked model changed (also fires once models are loaded) */
  onModelChange?: (id: string | null) => void;
}

export interface ChatHandle {
  /** read-only model label in the header */
  model: HTMLElement;
  /** fill the picker from GET /models (null: hide it) */
  setModels(list: ModelList | null): void;
  selected(): string | null;
}

const FOLLOW_SLACK_PX = 48; // within this of the bottom still counts as "following the stream"

/** DOM wiring for the chat panel. */
export function mountChat(root: HTMLElement, opts: ChatOptions): ChatHandle {
  const {
    onHighlight, onActivity = () => undefined, getPointCount = () => null,
    onStreaming = () => undefined, onModelChange = () => undefined,
  } = opts;
  root.innerHTML =
    '<header id="chat-head"><span id="agent-label">agent</span><span id="model"></span></header>' +
    '<div id="log" role="log" aria-live="polite" aria-label="conversation"></div>' +
    '<form id="form"><div id="picker" hidden><label for="model-select">model</label>' +
    '<select id="model-select" aria-label="Model for the next message"></select></div>' +
    '<div id="field"><input id="input" autocomplete="off" aria-label="Ask about the fly brain" placeholder="Ask about the fly brain\u2026" />' +
    '<button id="send" type="submit" aria-label="send"><svg viewBox="0 0 13 13" width="15" height="15" aria-hidden="true">' +
    `<path d="${ARROW_UP}" fill="currentColor"/></svg></button></div></form>`;
  const log = root.querySelector<HTMLElement>("#log")!;
  const form = root.querySelector<HTMLFormElement>("#form")!;
  const input = root.querySelector<HTMLInputElement>("#input")!;
  const send = root.querySelector<HTMLButtonElement>("#send")!;
  const picker = root.querySelector<HTMLElement>("#picker")!;
  const select = root.querySelector<HTMLSelectElement>("#model-select")!;
  let conversationId: string | null = null;
  let chosen: string | null = null;
  let following = true; // keep the log pinned to the bottom unless the user scrolled up

  log.addEventListener("scroll", () => {
    following = log.scrollHeight - log.scrollTop - log.clientHeight <= FOLLOW_SLACK_PX;
  });
  const follow = () => {
    if (following) log.scrollTop = log.scrollHeight;
  };

  select.addEventListener("change", () => {
    chosen = select.value || null;
    if (chosen) saveStoredModel(chosen);
    onModelChange(chosen);
  });

  const hint = document.createElement("button");
  hint.type = "button";
  hint.id = "hint";
  hint.textContent = HINT;
  hint.addEventListener("click", () => {
    input.value = HINT.replace(/^Try:\s*/, "");
    input.focus();
  });
  log.appendChild(hint);

  const add = (cls: string, text: string): HTMLElement => {
    hint.remove();
    const el = document.createElement("div");
    el.className = `msg ${cls}`;
    el.textContent = text;
    log.appendChild(el);
    follow();
    return el;
  };

  // live tool lines: running (pulsing dot, counting up) -> done / failed
  interface Tool { el: HTMLElement; stat: HTMLElement; t0: number; finished: boolean }
  const tools = new Map<string, Tool>();
  let anon = 0;
  const tickTools = () => {
    const now = performance.now();
    for (const t of tools.values()) if (!t.finished) t.stat.textContent = toolStatusText({ state: "running" }, now - t.t0);
  };
  const finishTool = (t: Tool, state: "done" | "failed", ms: number) => {
    t.finished = true;
    t.el.dataset.state = state;
    t.stat.textContent = toolStatusText({ state, ms }, 0);
  };

  // streamed answer: plain deltas re-rendered with the safe markdown renderer, once per frame
  let streamEl: HTMLElement | null = null;
  let streamText = "";
  let frame = 0;
  const paintStream = () => {
    frame = 0;
    if (streamEl) {
      streamEl.replaceChildren(renderMarkdown(streamText));
      follow();
    }
  };
  const dropStream = () => {
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
    streamEl?.remove();
    streamEl = null;
    streamText = "";
  };

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    input.disabled = send.disabled = true;
    following = true;
    add("user", text);
    onStreaming(true);
    // pending indicator: visible at once, until the first real event, so a slow provider never looks dead
    const started = performance.now();
    let thinking: HTMLElement | null = add("thinking", formatThinking(0));
    const tick = window.setInterval(() => {
      if (thinking) thinking.textContent = formatThinking((performance.now() - started) / 1000);
      tickTools();
    }, 1000);
    const settle = () => {
      thinking?.remove();
      thinking = null;
    };
    void streamChat(text, conversationId, {
      onMeta: (id) => { conversationId = id; },
      onToolCall: (tool, args, id) => {
        settle();
        dropStream(); // text before a tool call was a preamble, not the answer
        const el = add("tool", "");
        el.dataset.state = "running";
        const dot = document.createElement("span");
        dot.className = "dot";
        dot.setAttribute("aria-hidden", "true");
        const call = document.createElement("span");
        call.className = "call";
        call.textContent = formatToolCall(tool, args);
        const stat = document.createElement("span");
        stat.className = "tstat";
        stat.textContent = toolStatusText({ state: "running" }, 0);
        el.append(dot, call, stat);
        tools.set(id ?? `anon-${anon++}`, { el, stat, t0: performance.now(), finished: false });
      },
      onToolResult: (id, ok, ms) => {
        const t = tools.get(id);
        if (t && !t.finished) finishTool(t, ok ? "done" : "failed", ms);
      },
      onHighlight: (p) => {
        settle();
        const el = add("hl", "");
        const lit = document.createElement("span");
        lit.textContent = `\u25CF lit ${p.total.toLocaleString("en-US")} neurons`;
        const sel = document.createElement("span");
        sel.className = "sel";
        sel.textContent = p.selection_id;
        el.append(lit, sel);
        void Promise.resolve(onHighlight(p)).catch((err) => add("error", `highlight failed: ${(err as Error).message}`));
      },
      onActivity: (a) => {
        settle();
        const el = add("hl", "");
        const wave = document.createElement("span");
        wave.textContent = `\u25B6 activity \u00b7 ${(a.nBins * a.binMs) / 1000} s simulated`;
        const sel = document.createElement("span");
        sel.className = "sel";
        sel.textContent = a.simulationId;
        el.append(wave, sel);
        void Promise.resolve(onActivity(a)).catch((err) => add("error", `activity failed: ${(err as Error).message}`));
      },
      onAnswerDelta: (t) => {
        settle();
        if (!streamEl) streamEl = add("answer", "");
        streamText += t;
        if (!frame) frame = requestAnimationFrame(paintStream);
      },
      onAnswer: (t) => {
        settle();
        const el = streamEl ?? add("answer", "");
        if (frame) cancelAnimationFrame(frame);
        frame = 0;
        streamEl = null;
        streamText = "";
        el.replaceChildren(renderMarkdown(t));
        follow();
      },
      onError: (m) => { settle(); add("error", m); },
      onDone: () => {
        window.clearInterval(tick);
        settle();
        if (streamEl) paintStream(); // an interrupted stream keeps what arrived
        streamEl = null;
        for (const t of tools.values()) if (!t.finished) { t.finished = true; t.el.dataset.state = "ended"; t.stat.textContent = ""; }
        tools.clear();
        input.disabled = send.disabled = false;
        onStreaming(false);
        input.focus();
      },
    }, fetch, "/api", getPointCount, chosen);
  });

  return {
    model: root.querySelector<HTMLElement>("#model")!,
    selected: () => chosen,
    setModels(list) {
      select.replaceChildren();
      if (list === null || list.models.length === 0) {
        picker.hidden = true;
        chosen = null;
        onModelChange(null);
        return;
      }
      for (const m of list.models) {
        const o = document.createElement("option");
        o.value = m.id;
        o.textContent = optionLabel(m);
        o.disabled = !m.available;
        if (m.reason) o.title = m.reason;
        select.appendChild(o);
      }
      chosen = pickModel(list, loadStoredModel());
      if (chosen) select.value = chosen;
      picker.hidden = false;
      onModelChange(chosen);
    },
  };
}
