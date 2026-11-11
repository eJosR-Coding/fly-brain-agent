// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { mountChat } from "../src/chat";
import { EXPANDED_KEY } from "../src/toolTrace";

const sse = (events: string[]) => {
  const enc = new TextEncoder();
  return new Response(
    new ReadableStream<Uint8Array>({ start(c) { for (const e of events) c.enqueue(enc.encode(e)); c.close(); } }),
    { status: 200, headers: { "content-type": "text/event-stream" } },
  );
};

async function runTurn() {
  const root = document.createElement("div");
  document.body.appendChild(root);
  mountChat(root, { onHighlight: () => undefined });
  vi.stubGlobal("fetch", async () => sse([
    'event: tool_call\ndata: {"tool":"search_neurons","args":{"cell_type":"EPG"},"id":"t1"}\n\n',
    'event: tool_result\ndata: {"id":"t1","ok":true,"ms":1200}\n\n',
    'event: answer\ndata: {"text":"done"}\n\n',
    'event: done\ndata: {}\n\n',
  ]));
  (root.querySelector("#input") as HTMLInputElement).value = "hi";
  root.querySelector("#form")!.dispatchEvent(new Event("submit", { cancelable: true }));
  await vi.waitFor(() => expect((root.querySelector("#input") as HTMLInputElement).disabled).toBe(false));
  return root;
}

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
  document.body.replaceChildren();
});

describe("the collapsible tool trace in the chat", () => {
  it("renders one trace block per turn, collapsed by default, with the summary row", async () => {
    const root = await runTurn();
    const traces = root.querySelectorAll<HTMLElement>(".msg.trace");
    expect(traces).toHaveLength(1);
    const btn = traces[0].querySelector<HTMLButtonElement>("button.trace-sum")!;
    expect(traces[0].dataset.open).toBe("0");
    expect(btn.getAttribute("aria-expanded")).toBe("false");
    expect(traces[0].querySelector(".trace-body")!.hasAttribute("inert")).toBe(true);
    expect(btn.querySelector(".trace-label")!.textContent).toBe("Used 1 tool · 1.2 s");
    expect(btn.getAttribute("aria-controls")).toBe(traces[0].querySelector(".trace-body")!.id);
  });

  it("toggles open and closed on click and remembers the choice", async () => {
    const root = await runTurn();
    const trace = root.querySelector<HTMLElement>(".msg.trace")!;
    const btn = trace.querySelector<HTMLButtonElement>("button.trace-sum")!;
    btn.click();
    expect(trace.dataset.open).toBe("1");
    expect(btn.getAttribute("aria-expanded")).toBe("true");
    expect(trace.querySelector(".trace-body")!.hasAttribute("inert")).toBe(false);
    expect(localStorage.getItem(EXPANDED_KEY)).toBe("1");
    btn.click();
    expect(trace.dataset.open).toBe("0");
    expect(btn.getAttribute("aria-expanded")).toBe("false");
    expect(localStorage.getItem(EXPANDED_KEY)).toBe("0");
  });

  it("opens a new turn's trace expanded when the user left it open", async () => {
    localStorage.setItem(EXPANDED_KEY, "1");
    const root = await runTurn();
    expect(root.querySelector<HTMLElement>(".msg.trace")!.dataset.open).toBe("1");
  });
});
