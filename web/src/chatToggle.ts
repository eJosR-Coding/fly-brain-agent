/** One-click hide/show for the agent chat. The chat DOM is only hidden by CSS, never torn down. */
import { loadCollapsed, saveCollapsed, type Store } from "./panelState";

export const CLASS = "chat-collapsed";
export const LABEL_HIDE = "Hide the agent chat";
export const LABEL_SHOW = "Show the agent chat";

export interface ChatToggle {
  toggle(): void;
  collapsed(): boolean;
  destroy(): void;
}

function button(id: string, text: string, controls: string): HTMLButtonElement {
  const b = document.createElement("button");
  b.type = "button";
  b.id = id;
  b.textContent = text;
  if (controls) b.setAttribute("aria-controls", controls);
  return b;
}

export function mountChatToggle(root: HTMLElement, chat: HTMLElement, opts: { store?: Store | null } = {}): ChatToggle {
  const store = opts.store;
  const hide = button("chat-hide", "hide", chat.id);
  const tab = button("chat-show", "agent", chat.id);
  const head = chat.querySelector<HTMLElement>("#chat-head") ?? chat;
  head.appendChild(hide);
  root.appendChild(tab);

  let state = store === undefined ? loadCollapsed() : loadCollapsed(store);
  const render = () => {
    root.classList.toggle(CLASS, state);
    chat.toggleAttribute("inert", state); // hidden content must not take focus
    hide.setAttribute("aria-expanded", String(!state));
    hide.setAttribute("aria-label", LABEL_HIDE);
    tab.setAttribute("aria-expanded", String(!state));
    tab.setAttribute("aria-label", LABEL_SHOW);
  };
  const set = (next: boolean) => {
    state = next;
    if (store === undefined) saveCollapsed(state);
    else saveCollapsed(state, store);
    render();
  };
  // Focus follows the control that is now visible, but only after the user clicked one: never at mount,
  // from a stored preference or from a programmatic toggle().
  const onHide = () => { set(true); tab.focus({ preventScroll: true }); };
  const onShow = () => { set(false); hide.focus({ preventScroll: true }); };
  hide.addEventListener("click", onHide);
  tab.addEventListener("click", onShow);
  render();

  return {
    toggle: () => set(!state),
    collapsed: () => state,
    destroy() {
      hide.removeEventListener("click", onHide);
      tab.removeEventListener("click", onShow);
      hide.remove();
      tab.remove();
      root.classList.remove(CLASS);
      chat.removeAttribute("inert");
    },
  };
}
