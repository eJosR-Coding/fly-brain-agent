/** Remembered open/closed state of the agent chat panel. Storage failures never break the page. */
export const COLLAPSED_KEY = "flybrain.chat.collapsed";
export type Store = Pick<Storage, "getItem" | "setItem">;

function defaultStore(): Store | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null; // blocked storage can throw on access
  }
}

/** Not collapsed unless the user hid it. */
export function loadCollapsed(store: Store | null = defaultStore()): boolean {
  try {
    return store?.getItem(COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

export function saveCollapsed(collapsed: boolean, store: Store | null = defaultStore()): void {
  try {
    store?.setItem(COLLAPSED_KEY, collapsed ? "1" : "0");
  } catch {
    /* private mode or blocked storage: the preference just is not remembered */
  }
}
