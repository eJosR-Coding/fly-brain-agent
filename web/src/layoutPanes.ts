/** The two stage panes (fly body, brain). Which one is large, and whether the inset shows, is a set of CSS classes on
 *  the root: swapping never moves or recreates a node, so the scenes inside keep running (they follow their own
 *  ResizeObserver). A hidden inset is visibility:hidden, never display:none, so its scene never sees a zero size. */
import { loadLayout, saveLayout, swap, toggleInset, type LayoutState, type Pane, type Store } from "./layoutState";

export const MAIN_CLASS: Record<Pane, string> = { body: "layout-main-body", brain: "layout-main-brain" };
export const HIDDEN_CLASS = "inset-hidden";

export interface Layout {
  /** containers for the scenes: mount the body scene and the brain preview/scene here */
  bodyHost: HTMLElement;
  brainHost: HTMLElement;
  state(): LayoutState;
  swap(): void;
  toggleInset(): void;
  destroy(): void;
}

export interface LayoutOptions {
  store?: Store | null;
  /** after every user change (not the initial render) */
  onChange?: (state: LayoutState) => void;
}

const other = (p: Pane): Pane => (p === "body" ? "brain" : "body");

function button(cls: string, text: string): HTMLButtonElement {
  const b = document.createElement("button");
  b.type = "button";
  b.className = cls;
  b.textContent = text;
  return b;
}

interface PaneParts {
  pane: HTMLElement;
  head: HTMLElement;
  swapBtn: HTMLButtonElement;
  hideBtn: HTMLButtonElement;
  host: HTMLElement;
}

function buildPane(name: Pane): PaneParts {
  const pane = document.createElement("section");
  pane.id = `pane-${name}`;
  pane.className = "pane";
  const head = document.createElement("div");
  head.className = "pane-head";
  const label = document.createElement("span");
  label.className = "pane-label";
  label.textContent = name;
  const swapBtn = button("pane-swap", "swap");
  swapBtn.setAttribute("aria-label", `Swap: show the ${name} large`);
  const hideBtn = button("pane-hide", "hide");
  hideBtn.setAttribute("aria-label", `Hide the ${name} panel`);
  hideBtn.setAttribute("aria-controls", pane.id);
  head.append(label, swapBtn, hideBtn);
  const host = document.createElement("div");
  host.className = "pane-view";
  pane.append(head, host);
  return { pane, head, swapBtn, hideBtn, host };
}

export function mountLayout(root: HTMLElement, opts: LayoutOptions = {}): Layout {
  const store = opts.store === undefined ? undefined : opts.store;
  let state = store === undefined ? loadLayout() : loadLayout(store);
  const parts: Record<Pane, PaneParts> = { body: buildPane("body"), brain: buildPane("brain") };
  const tab = button("pane-show", "");
  tab.id = "pane-show";
  tab.setAttribute("aria-controls", parts.brain.pane.id);
  root.append(parts.body.pane, parts.brain.pane, tab);

  const render = () => {
    const inset = other(state.main);
    for (const m of ["body", "brain"] as const) root.classList.toggle(MAIN_CLASS[m], state.main === m);
    root.classList.toggle(HIDDEN_CLASS, state.insetHidden);
    for (const n of ["body", "brain"] as const) {
      const p = parts[n];
      p.head.hidden = n !== inset; // only the inset has a header
      p.hideBtn.setAttribute("aria-expanded", String(!state.insetHidden));
      p.pane.toggleAttribute("inert", n === inset && state.insetHidden); // a hidden inset must not take focus
    }
    tab.hidden = !state.insetHidden;
    tab.textContent = inset;
    tab.setAttribute("aria-label", `Show the ${inset} panel`);
    tab.setAttribute("aria-controls", parts[inset].pane.id);
    tab.setAttribute("aria-expanded", String(!state.insetHidden));
  };
  const commit = (next: LayoutState, focus: HTMLElement) => {
    state = next;
    if (store === undefined) saveLayout(state);
    else saveLayout(state, store);
    render();
    focus.focus({ preventScroll: true }); // the control that was clicked may be gone: focus the visible one
    opts.onChange?.(state);
  };
  const insetParts = () => parts[other(state.main)];
  const onSwap = () => {
    const next = swap(state);
    commit(next, parts[other(next.main)].swapBtn);
  };
  const onHide = () => commit(toggleInset(state), tab);
  const onShow = () => commit(toggleInset(state), insetParts().hideBtn);
  const listeners: [HTMLElement, () => void][] = [
    [parts.body.swapBtn, onSwap], [parts.brain.swapBtn, onSwap],
    [parts.body.hideBtn, onHide], [parts.brain.hideBtn, onHide],
    [tab, onShow],
  ];
  for (const [el, fn] of listeners) el.addEventListener("click", fn);
  render();

  return {
    bodyHost: parts.body.host,
    brainHost: parts.brain.host,
    state: () => ({ ...state }),
    swap: onSwap,
    toggleInset: () => (state.insetHidden ? onShow() : onHide()),
    destroy() {
      for (const [el, fn] of listeners) el.removeEventListener("click", fn);
      parts.body.pane.remove();
      parts.brain.pane.remove();
      tab.remove();
      root.classList.remove(MAIN_CLASS.body, MAIN_CLASS.brain, HIDDEN_CLASS);
    },
  };
}
