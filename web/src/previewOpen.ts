/** The default-mode control over the brain preview: a focusable "button" that opens the real scene on demand. */
import { isActivationKey, isZoomIntent, type BrainGate, type GateState } from "./brainGate";

export const LABEL = "Load the neurons (zoom in, click or press Enter)";
export const HINT_IDLE = "zoom or click to load the neurons";
export const HINT_LOADING = "loading neurons...";
export const HINT_FAILED = "could not load the neurons, click to retry";

export interface OpenControl {
  remove(): void;
}

export function mountOpenControl(host: HTMLElement, gate: BrainGate<unknown>): OpenControl {
  const el = document.createElement("div");
  el.className = "brain-open";
  el.setAttribute("role", "button");
  el.tabIndex = 0;
  el.setAttribute("aria-label", LABEL);
  const hint = document.createElement("span");
  hint.className = "brain-hint";
  // The aria-label already says it: the visible hint is hidden from assistive tech so it is not announced twice.
  hint.setAttribute("aria-hidden", "true");
  el.appendChild(hint);
  host.appendChild(el);

  const render = (s: GateState) => {
    hint.textContent = s === "loading" ? HINT_LOADING : s === "failed" ? HINT_FAILED : HINT_IDLE;
    // Only the failure changes what a screen reader should hear; the loading text is not announced.
    el.setAttribute("aria-label", s === "failed" ? HINT_FAILED : LABEL);
    el.setAttribute("aria-busy", s === "loading" ? "true" : "false");
  };
  const open = () => {
    if (gate.state() === "loading" || gate.state() === "ready") return;
    void gate.ensure().catch(() => undefined); // failure is shown through the gate state
  };
  const onWheel = (e: WheelEvent) => {
    if (gate.state() !== "failed" && isZoomIntent(e)) open(); // a wheel never hammers a failed load
  };
  const onKey = (e: KeyboardEvent) => {
    if (!isActivationKey(e.key)) return;
    e.preventDefault(); // Space must not scroll the page
    open();
  };
  el.addEventListener("wheel", onWheel, { passive: true });
  el.addEventListener("click", open);
  el.addEventListener("keydown", onKey);
  const off = gate.subscribe(render);
  render(gate.state());
  return {
    remove() {
      off();
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("click", open);
      el.removeEventListener("keydown", onKey);
      el.remove();
    },
  };
}
