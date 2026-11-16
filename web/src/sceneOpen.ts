/** When the layout asks for the real brain scene. Pure. */
import type { LayoutState, Pane } from "./layoutState";

/** The layout never reports its initial state, so a remembered "brain large" must open the scene at boot. */
export const shouldOpenOnBoot = (s: LayoutState): boolean => s.main === "brain";

/** Only a change of the main pane opens it: hiding/showing the inset must not retry a failed load. */
export const shouldOpenOnChange = (previousMain: Pane, next: LayoutState): boolean =>
  next.main === "brain" && previousMain !== "brain";
