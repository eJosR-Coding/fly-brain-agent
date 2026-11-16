// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";
import { mountChatToggle } from "../src/chatToggle";
import { COLLAPSED_KEY } from "../src/panelState";

function memory(initial: Record<string, string> = {}) {
  const data = { ...initial };
  return { data, getItem: (k: string) => data[k] ?? null, setItem: (k: string, v: string) => void (data[k] = v) };
}

function setup(initial: Record<string, string> = {}) {
  const root = document.createElement("div");
  const chat = document.createElement("aside");
  chat.id = "chat";
  chat.innerHTML = '<header id="chat-head"><span id="agent-label">agent</span></header><div id="log"><p id="msg">hello</p></div>';
  root.appendChild(chat);
  document.body.appendChild(root);
  const store = memory(initial);
  const toggle = mountChatToggle(root, chat, { store });
  const hide = chat.querySelector<HTMLButtonElement>("#chat-hide")!;
  const tab = root.querySelector<HTMLButtonElement>("#chat-show")!;
  return { root, chat, store, toggle, hide, tab };
}

describe("mountChatToggle focus", () => {
  const spy = () => {
    const root = document.createElement("div");
    const chat = document.createElement("aside");
    chat.id = "chat";
    chat.innerHTML = '<header id="chat-head"></header>';
    root.appendChild(chat);
    document.body.appendChild(root);
    return { root, chat };
  };

  it("does not move focus at mount, expanded or collapsed from a stored preference", () => {
    const focus = vi.spyOn(HTMLElement.prototype, "focus");
    for (const initial of [{}, { [COLLAPSED_KEY]: "1" }]) {
      const { root, chat } = spy();
      mountChatToggle(root, chat, { store: memory(initial) });
    }
    expect(focus).not.toHaveBeenCalled();
    focus.mockRestore();
  });

  it("moves focus to the visible control only after a user click", () => {
    const { hide, tab } = setup();
    const focus = vi.spyOn(HTMLElement.prototype, "focus");
    hide.click();
    expect(focus).toHaveBeenCalledTimes(1);
    expect(focus.mock.contexts[0]).toBe(tab);
    tab.click();
    expect(focus).toHaveBeenCalledTimes(2);
    expect(focus.mock.contexts[1]).toBe(hide);
    focus.mockRestore();
  });

  it("toggle() called from code changes the state without stealing focus", () => {
    const { toggle } = setup();
    const focus = vi.spyOn(HTMLElement.prototype, "focus");
    toggle.toggle();
    expect(toggle.collapsed()).toBe(true);
    expect(focus).not.toHaveBeenCalled();
    focus.mockRestore();
  });
});

describe("mountChatToggle", () => {
  it("starts expanded by default with the hide button wired to the chat", () => {
    const { root, hide, chat, toggle } = setup();
    expect(toggle.collapsed()).toBe(false);
    expect(root.classList.contains("chat-collapsed")).toBe(false);
    expect(hide.getAttribute("aria-expanded")).toBe("true");
    expect(hide.getAttribute("aria-controls")).toBe(chat.id);
    expect(hide.getAttribute("aria-label")).toBe("Hide the agent chat");
    expect(hide.querySelector("svg, img")).toBeNull();
  });

  it("starts collapsed when the stored preference says so", () => {
    const { root, tab, toggle } = setup({ [COLLAPSED_KEY]: "1" });
    expect(toggle.collapsed()).toBe(true);
    expect(root.classList.contains("chat-collapsed")).toBe(true);
    expect(tab.getAttribute("aria-expanded")).toBe("false");
    expect(tab.getAttribute("aria-label")).toBe("Show the agent chat");
  });

  it("a click on hide collapses, persists, and the show tab restores", () => {
    const { root, hide, tab, store, chat } = setup();
    hide.click();
    expect(root.classList.contains("chat-collapsed")).toBe(true);
    expect(store.data[COLLAPSED_KEY]).toBe("1");
    expect(hide.getAttribute("aria-expanded")).toBe("false");
    expect(tab.getAttribute("aria-expanded")).toBe("false");
    expect(chat.hasAttribute("inert")).toBe(true);
    tab.click();
    expect(root.classList.contains("chat-collapsed")).toBe(false);
    expect(store.data[COLLAPSED_KEY]).toBe("0");
    expect(hide.getAttribute("aria-expanded")).toBe("true");
    expect(chat.hasAttribute("inert")).toBe(false);
  });

  it("toggle() flips the state", () => {
    const { toggle, root } = setup();
    toggle.toggle();
    expect(toggle.collapsed()).toBe(true);
    toggle.toggle();
    expect(toggle.collapsed()).toBe(false);
    expect(root.classList.contains("chat-collapsed")).toBe(false);
  });

  it("destroy removes the controls and listeners and leaves the conversation DOM intact", () => {
    const { root, hide, chat, toggle } = setup();
    const msg = chat.querySelector("#msg");
    toggle.destroy();
    expect(root.querySelector("#chat-show")).toBeNull();
    expect(chat.querySelector("#chat-hide")).toBeNull();
    expect(chat.querySelector("#msg")).toBe(msg);
    hide.click(); // detached button no longer drives anything
    expect(root.classList.contains("chat-collapsed")).toBe(false);
  });

  it("keeps working when storage throws", () => {
    const root = document.createElement("div");
    const chat = document.createElement("aside");
    chat.innerHTML = '<header id="chat-head"></header>';
    root.appendChild(chat);
    const bad = { getItem: () => { throw new Error("x"); }, setItem: () => { throw new Error("x"); } };
    const t = mountChatToggle(root, chat, { store: bad });
    t.toggle();
    expect(t.collapsed()).toBe(true);
  });
});
