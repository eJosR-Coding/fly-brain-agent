// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";
import { mountLayout } from "../src/layoutPanes";
import { LAYOUT_KEY } from "../src/layoutState";

function memory(initial: Record<string, string> = {}) {
  const data = { ...initial };
  return { data, getItem: (k: string) => data[k] ?? null, setItem: (k: string, v: string) => void (data[k] = v) };
}

function setup(initial: Record<string, string> = {}, extra: { onChange?: (s: { main: string; insetHidden: boolean }) => void } = {}) {
  const root = document.createElement("div");
  root.id = "stage";
  document.body.appendChild(root);
  const store = memory(initial);
  const layout = mountLayout(root, { store, ...extra });
  const q = <T extends HTMLElement>(sel: string) => root.querySelector<T>(sel)!;
  return { root, store, layout, q };
}
const stored = (v: object) => ({ [LAYOUT_KEY]: JSON.stringify(v) });

describe("mountLayout", () => {
  it("builds the two panes and starts with the body large by default", () => {
    const { root, layout, q } = setup();
    expect(root.classList.contains("layout-main-body")).toBe(true);
    expect(root.classList.contains("inset-hidden")).toBe(false);
    expect(q("#pane-body")).toBeTruthy();
    expect(q("#pane-brain")).toBeTruthy();
    expect(layout.bodyHost.closest("#pane-body")).toBe(q("#pane-body"));
    expect(layout.brainHost.closest("#pane-brain")).toBe(q("#pane-brain"));
    expect(q("#pane-show").hidden).toBe(true);
  });

  it("reads the initial classes from storage", () => {
    const { root, q } = setup(stored({ main: "brain", insetHidden: true }));
    expect(root.classList.contains("layout-main-brain")).toBe(true);
    expect(root.classList.contains("inset-hidden")).toBe(true);
    expect(q("#pane-show").hidden).toBe(false);
  });

  it("the brain swap button flips the classes, labels and persists", () => {
    const { root, store, q } = setup();
    const brainSwap = q<HTMLButtonElement>("#pane-brain .pane-swap");
    const bodySwap = q<HTMLButtonElement>("#pane-body .pane-swap");
    expect(brainSwap.getAttribute("aria-label")).toBe("Swap: show the brain large");
    expect(bodySwap.getAttribute("aria-label")).toBe("Swap: show the body large");
    brainSwap.click();
    expect(root.classList.contains("layout-main-brain")).toBe(true);
    expect(root.classList.contains("layout-main-body")).toBe(false);
    expect(JSON.parse(store.data[LAYOUT_KEY])).toEqual({ main: "brain", insetHidden: false });
    bodySwap.click();
    expect(root.classList.contains("layout-main-body")).toBe(true);
    expect(JSON.parse(store.data[LAYOUT_KEY])).toEqual({ main: "body", insetHidden: false });
  });

  it("only the inset pane header is shown to assistive tech and the other is hidden", () => {
    const { q } = setup();
    expect(q("#pane-brain .pane-head").hidden).toBe(false);
    expect(q("#pane-body .pane-head").hidden).toBe(true);
    q<HTMLButtonElement>("#pane-brain .pane-swap").click();
    expect(q("#pane-brain .pane-head").hidden).toBe(true);
    expect(q("#pane-body .pane-head").hidden).toBe(false);
  });

  it("labels the inset header and the show tab with the inset pane", () => {
    const { q } = setup();
    expect(q("#pane-brain .pane-label").textContent).toBe("brain");
    expect(q("#pane-brain .pane-hide").getAttribute("aria-label")).toBe("Hide the brain panel");
    q<HTMLButtonElement>("#pane-brain .pane-hide").click();
    expect(q("#pane-show").textContent).toBe("brain");
    expect(q("#pane-show").getAttribute("aria-label")).toBe("Show the brain panel");
    q<HTMLButtonElement>("#pane-show").click();
    q<HTMLButtonElement>("#pane-brain .pane-swap").click(); // now the body is the inset
    q<HTMLButtonElement>("#pane-body .pane-hide").click();
    expect(q("#pane-show").textContent).toBe("body");
    expect(q("#pane-show").getAttribute("aria-label")).toBe("Show the body panel");
    expect(q("#pane-body .pane-hide").getAttribute("aria-label")).toBe("Hide the body panel");
  });

  it("hide hides the inset, shows the tab and persists; the tab restores it", () => {
    const { root, store, q } = setup();
    const hide = q<HTMLButtonElement>("#pane-brain .pane-hide");
    const tab = q<HTMLButtonElement>("#pane-show");
    expect(hide.getAttribute("aria-expanded")).toBe("true");
    hide.click();
    expect(root.classList.contains("inset-hidden")).toBe(true);
    expect(tab.hidden).toBe(false);
    expect(tab.getAttribute("aria-expanded")).toBe("false");
    expect(JSON.parse(store.data[LAYOUT_KEY])).toEqual({ main: "body", insetHidden: true });
    tab.click();
    expect(root.classList.contains("inset-hidden")).toBe(false);
    expect(tab.hidden).toBe(true);
    expect(JSON.parse(store.data[LAYOUT_KEY])).toEqual({ main: "body", insetHidden: false });
  });

  it("moves focus to the control that is now visible", () => {
    const { q } = setup();
    q<HTMLButtonElement>("#pane-brain .pane-hide").click();
    expect(document.activeElement).toBe(q("#pane-show"));
    q<HTMLButtonElement>("#pane-show").click();
    expect(document.activeElement).toBe(q("#pane-brain .pane-hide"));
  });

  it("swapping keeps the very same DOM nodes (no recreation, no moves)", () => {
    const { q, layout } = setup();
    const body = q("#pane-body"), brain = q("#pane-brain");
    const bodyHost = layout.bodyHost, brainHost = layout.brainHost;
    const marker = document.createElement("canvas");
    brainHost.appendChild(marker); // stands for the scene's canvas
    q<HTMLButtonElement>("#pane-brain .pane-swap").click();
    q<HTMLButtonElement>("#pane-body .pane-swap").click();
    q<HTMLButtonElement>("#pane-brain .pane-hide").click();
    q<HTMLButtonElement>("#pane-show").click();
    expect(q("#pane-body")).toBe(body);
    expect(q("#pane-brain")).toBe(brain);
    expect(layout.bodyHost).toBe(bodyHost);
    expect(layout.brainHost).toBe(brainHost);
    expect(marker.parentElement).toBe(brainHost);
    expect(marker.isConnected).toBe(true);
  });

  it("calls onChange with the new state for swap and hide, not for the initial render", () => {
    const onChange = vi.fn();
    const { q } = setup({}, { onChange });
    expect(onChange).not.toHaveBeenCalled();
    q<HTMLButtonElement>("#pane-brain .pane-swap").click();
    expect(onChange).toHaveBeenLastCalledWith({ main: "brain", insetHidden: false });
    q<HTMLButtonElement>("#pane-body .pane-hide").click();
    expect(onChange).toHaveBeenLastCalledWith({ main: "brain", insetHidden: true });
  });

  it("exposes the state and programmatic swap/toggle", () => {
    const { root, layout } = setup();
    layout.swap();
    expect(layout.state()).toEqual({ main: "brain", insetHidden: false });
    layout.toggleInset();
    expect(layout.state().insetHidden).toBe(true);
    expect(root.classList.contains("inset-hidden")).toBe(true);
  });

  it("works when storage throws", () => {
    const root = document.createElement("div");
    document.body.appendChild(root);
    const bad = { getItem: () => { throw new Error("x"); }, setItem: () => { throw new Error("x"); } };
    const layout = mountLayout(root, { store: bad });
    expect(root.classList.contains("layout-main-body")).toBe(true);
    expect(() => layout.swap()).not.toThrow();
    expect(root.classList.contains("layout-main-brain")).toBe(true);
  });

  it("destroy removes the controls and classes and stops reacting to clicks", () => {
    const { root, layout, q } = setup();
    const swapBtn = q<HTMLButtonElement>("#pane-brain .pane-swap");
    const bodyHost = layout.bodyHost;
    layout.destroy();
    expect(root.querySelector("#pane-show")).toBeNull();
    expect(root.classList.contains("layout-main-body")).toBe(false);
    expect(root.classList.contains("inset-hidden")).toBe(false);
    expect(root.querySelector("#pane-body")).toBeNull();
    swapBtn.click();
    expect(root.classList.contains("layout-main-brain")).toBe(false);
    expect(bodyHost.isConnected).toBe(false);
  });
});
