/** Layout of the stage: which of the two scenes is large, and whether the inset is shown. Pure, plus a safe store. */
export const LAYOUT_KEY = "flybrain.layout";
export type Pane = "body" | "brain";
export interface LayoutState {
  main: Pane;
  insetHidden: boolean;
}
export type Store = Pick<Storage, "getItem" | "setItem">;

export const DEFAULT_LAYOUT: Readonly<LayoutState> = Object.freeze({ main: "body", insetHidden: false });

export const swap = (s: LayoutState): LayoutState => ({ ...s, main: s.main === "body" ? "brain" : "body" });
export const toggleInset = (s: LayoutState): LayoutState => ({ ...s, insetHidden: !s.insetHidden });

/** Validate a stored value (an object or its JSON text). A missing field takes its default; a wrong one voids it all. */
export function parseLayout(raw: unknown): LayoutState {
  let v = raw;
  if (typeof v === "string") {
    try {
      v = JSON.parse(v);
    } catch {
      return { ...DEFAULT_LAYOUT };
    }
  }
  if (typeof v !== "object" || v === null || Array.isArray(v)) return { ...DEFAULT_LAYOUT };
  const o = v as Record<string, unknown>;
  if (o.main !== undefined && o.main !== "body" && o.main !== "brain") return { ...DEFAULT_LAYOUT };
  if (o.insetHidden !== undefined && typeof o.insetHidden !== "boolean") return { ...DEFAULT_LAYOUT };
  return {
    main: (o.main as Pane | undefined) ?? DEFAULT_LAYOUT.main,
    insetHidden: (o.insetHidden as boolean | undefined) ?? DEFAULT_LAYOUT.insetHidden,
  };
}

function defaultStore(): Store | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null; // blocked storage can throw on access
  }
}

export function loadLayout(store: Store | null = defaultStore()): LayoutState {
  try {
    return parseLayout(store?.getItem(LAYOUT_KEY));
  } catch {
    return { ...DEFAULT_LAYOUT };
  }
}

export function saveLayout(state: LayoutState, store: Store | null = defaultStore()): void {
  try {
    store?.setItem(LAYOUT_KEY, JSON.stringify(state));
  } catch {
    /* private mode or blocked storage: the layout just is not remembered */
  }
}
