// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { mountChat, type ChatOptions } from "../src/chat";

vi.mock("../src/markdown", async (orig) => {
  const real = await orig<typeof import("../src/markdown")>();
  return { ...real, renderMarkdown: (t: string) => { if (t.includes("BOOM")) throw new Error("render exploded"); return real.renderMarkdown(t); } };
});

const enc = new TextEncoder();
/** A response whose body the test controls; it stays open until `close()`. */
function controlled(signal?: AbortSignal | null) {
  let ctrl!: ReadableStreamDefaultController<Uint8Array>;
  const body = new ReadableStream<Uint8Array>({ start(c) { ctrl = c; } });
  signal?.addEventListener("abort", () => { try { ctrl.error(new DOMException("aborted", "AbortError")); } catch { /* closed */ } });
  return {
    res: new Response(body, { status: 200 }),
    push: (s: string) => ctrl.enqueue(enc.encode(s)),
    close: () => ctrl.close(),
  };
}

function mount(opts: Partial<ChatOptions> = {}) {
  const root = document.createElement("div");
  document.body.appendChild(root);
  const handle = mountChat(root, { onHighlight: () => undefined, ...opts });
  const q = <T extends HTMLElement>(s: string) => root.querySelector<T>(s)!;
  const submit = (text: string) => {
    q<HTMLInputElement>("#input").value = text;
    q("#form").dispatchEvent(new Event("submit", { cancelable: true }));
  };
  return { root, handle, q, submit };
}

afterEach(() => {
  vi.unstubAllGlobals();
  document.body.replaceChildren();
});

describe("chat turn lifecycle", () => {
  it("a stalled stream can be stopped: send becomes stop, abort re-enables everything, onDone runs once", async () => {
    const streaming: boolean[] = [];
    const { q, submit } = mount({ onStreaming: (s) => streaming.push(s) });
    vi.stubGlobal("fetch", async (_u: string, init: RequestInit) => {
      const c = controlled(init.signal);
      c.push('event: meta\ndata: {"conversation_id":"c1"}\n\n');
      return c.res;
    });
    submit("hi");
    await vi.waitFor(() => expect(q("#send").getAttribute("aria-label")).toBe("stop"));
    expect(q<HTMLInputElement>("#input").disabled).toBe(true);
    expect(q<HTMLButtonElement>("#send").disabled).toBe(false);
    expect(q<HTMLButtonElement>("#new-chat").disabled).toBe(true);
    q("#form").dispatchEvent(new Event("submit", { cancelable: true })); // the stop press
    await vi.waitFor(() => expect(q<HTMLInputElement>("#input").disabled).toBe(false));
    expect(q("#send").getAttribute("aria-label")).toBe("send");
    expect(q<HTMLButtonElement>("#new-chat").disabled).toBe(false);
    expect(streaming).toEqual([true, false]);
    expect(q("#log").textContent).toMatch(/stopped/);
  });

  it("a second submit while a turn runs does not start another request", async () => {
    const { q, submit } = mount();
    const f = vi.fn(async (_u: string, init: RequestInit) => controlled(init.signal).res);
    vi.stubGlobal("fetch", f);
    submit("one");
    await vi.waitFor(() => expect(f).toHaveBeenCalledTimes(1));
    submit("two"); // while running this is the stop press, never a new turn
    await vi.waitFor(() => expect(q<HTMLInputElement>("#input").disabled).toBe(false));
    expect(f).toHaveBeenCalledTimes(1);
  });

  it("a markdown render failure while streaming is logged and falls back to plain text, never uncaught", async () => {
    const errors: string[] = [];
    const onError = (e: ErrorEvent) => { errors.push(e.message); e.preventDefault(); };
    window.addEventListener("error", onError);
    const log = vi.spyOn(console, "error").mockImplementation(() => {});
    const { q, submit } = mount();
    const c = controlled();
    vi.stubGlobal("fetch", async () => c.res);
    submit("hi");
    c.push('event: answer_delta\ndata: {"text":"BOOM"}\n\n');
    await vi.waitFor(() => expect(q("#log").textContent).toContain("BOOM"), { timeout: 2000 }); // the raw text is shown
    await new Promise((r) => setTimeout(r, 30));
    window.removeEventListener("error", onError);
    expect(errors).toEqual([]); // nothing escaped from the animation frame
    expect(log).toHaveBeenCalled();
    c.close();
    log.mockRestore();
  });

  it("input is re-enabled even when onDone's cleanup throws", async () => {
    const log = vi.spyOn(console, "error").mockImplementation(() => {});
    const { q, submit } = mount();
    vi.stubGlobal("fetch", async () => {
      const c = controlled();
      c.push('event: answer_delta\ndata: {"text":"BOOM"}\n\n');
      setTimeout(() => c.close(), 5);
      return c.res;
    });
    submit("hi");
    await vi.waitFor(() => expect(q<HTMLInputElement>("#input").disabled).toBe(false), { timeout: 2000 });
    expect(q<HTMLButtonElement>("#send").disabled).toBe(false);
    log.mockRestore();
  });

  it("events that arrive after done never touch the next turn", async () => {
    const { q, submit } = mount();
    const first = controlled();
    let n = 0;
    vi.stubGlobal("fetch", async (_u: string, init: RequestInit) => (++n === 1 ? first.res : controlled(init.signal).res));
    submit("one");
    await vi.waitFor(() => expect(n).toBe(1));
    first.push('event: done\ndata: {}\n\n'); // connection stays open
    await vi.waitFor(() => expect(q<HTMLInputElement>("#input").disabled).toBe(false));
    submit("two");
    await vi.waitFor(() => expect(n).toBe(2));
    first.push('event: answer\ndata: {"text":"STALE"}\n\n');
    await new Promise((r) => setTimeout(r, 20));
    expect(q("#log").textContent).not.toContain("STALE");
  });
});

describe("the model a turn started with", () => {
  const list = { default: "m1", models: [{ id: "m1", provider: "p", available: true }, { id: "m2", provider: "p", available: true }] };

  it("is what the status sees until the turn ends, even if the list loads mid-turn", async () => {
    const seen: (string | null)[] = [];
    const { q, handle, submit } = mount({ onModelChange: (id) => seen.push(id) });
    const bodies: string[] = [];
    let c!: ReturnType<typeof controlled>;
    vi.stubGlobal("fetch", async (_u: string, init: RequestInit) => {
      bodies.push(String(init.body));
      c = controlled(init.signal);
      return c.res;
    });
    submit("hi"); // list not loaded yet: sent with the server default
    await vi.waitFor(() => expect(bodies).toHaveLength(1));
    expect(JSON.parse(bodies[0]).model).toBeUndefined();
    seen.length = 0;
    handle.setModels(list);
    expect(seen).toEqual([null]); // the status keeps the running turn's model (server default), not the new m1
    expect(q<HTMLSelectElement>("#model-select").disabled).toBe(true);
    c.push('event: done\ndata: {}\n\n');
    await vi.waitFor(() => expect(q<HTMLInputElement>("#input").disabled).toBe(false));
    expect(seen).toEqual([null, "m1"]); // the picker choice shows again once the turn ended
    expect(q<HTMLSelectElement>("#model-select").disabled).toBe(false);
  });

  it("leaves the select empty when every model is unavailable, matching what is sent", () => {
    const { q, handle } = mount();
    handle.setModels({ default: null, models: [{ id: "x", provider: "p", available: false }] });
    expect(handle.selected()).toBeNull();
    expect(q<HTMLSelectElement>("#model-select").selectedIndex).toBe(-1);
  });
});
