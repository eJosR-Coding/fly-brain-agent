// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";
import { createBrainGate, type BrainGate } from "../src/brainGate";
import { HINT_FAILED, HINT_IDLE, HINT_LOADING, mountOpenControl } from "../src/previewOpen";

function setup(load: () => Promise<number> = async () => 1) {
  const host = document.createElement("div");
  const gate = createBrainGate({ load });
  const ensure = vi.spyOn(gate, "ensure");
  const control = mountOpenControl(host, gate as BrainGate<number>);
  const el = host.querySelector(".brain-open") as HTMLElement;
  const hint = host.querySelector(".brain-hint") as HTMLElement;
  return { host, gate, ensure, control, el, hint };
}

const wheel = (deltaY: number, ctrlKey = false) => new WheelEvent("wheel", { deltaY, ctrlKey, bubbles: true });

describe("mountOpenControl", () => {
  it("is a focusable button with a label and the idle hint", () => {
    const { el, hint } = setup();
    expect(el.getAttribute("role")).toBe("button");
    expect(el.tabIndex).toBe(0);
    expect(el.getAttribute("aria-label")).toBe("Load the neurons (zoom in, click or press Enter)");
    expect(hint.textContent).toBe(HINT_IDLE);
    expect(hint.getAttribute("aria-hidden")).toBe("true");
  });

  it("zoom-in wheel calls ensure once; zoom-out scroll does not", () => {
    const { el, ensure } = setup();
    el.dispatchEvent(wheel(20));
    expect(ensure).not.toHaveBeenCalled();
    el.dispatchEvent(wheel(-20));
    expect(ensure).toHaveBeenCalledTimes(1);
    el.dispatchEvent(wheel(-20)); // still loading: shared load, no second ensure
    expect(ensure).toHaveBeenCalledTimes(1);
  });

  it("click triggers ensure", () => {
    const { el, ensure } = setup();
    el.click();
    expect(ensure).toHaveBeenCalledTimes(1);
  });

  it("Enter and Space on the focused control trigger ensure; other keys do not", () => {
    const { el, ensure } = setup();
    el.dispatchEvent(new KeyboardEvent("keydown", { key: "a", bubbles: true }));
    expect(ensure).not.toHaveBeenCalled();
    const space = new KeyboardEvent("keydown", { key: " ", bubbles: true, cancelable: true });
    el.dispatchEvent(space);
    expect(space.defaultPrevented).toBe(true); // no page scroll
    expect(ensure).toHaveBeenCalledTimes(1);
    const enter = new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true });
    el.dispatchEvent(enter);
    expect(enter.defaultPrevented).toBe(true);
    expect(ensure).toHaveBeenCalledTimes(1); // already loading: the control does not re-enter
  });

  it("shows the loading hint while loading", () => {
    let release!: (n: number) => void;
    const { el, hint } = setup(() => new Promise<number>((r) => { release = r; }));
    el.click();
    expect(hint.textContent).toBe(HINT_LOADING);
    release(1);
  });

  it("shows the error hint on failure, no unhandled rejection, and click retries", async () => {
    const load = vi.fn().mockRejectedValueOnce(new Error("down")).mockResolvedValueOnce(2);
    const { el, hint, gate } = setup(load);
    el.click();
    await vi.waitFor(() => expect(gate.state()).toBe("failed"));
    expect(hint.textContent).toBe(HINT_FAILED);
    expect(el.getAttribute("aria-label")).toBe(HINT_FAILED);
    el.dispatchEvent(wheel(-20)); // a wheel must not hammer a failed load
    expect(load).toHaveBeenCalledTimes(1);
    el.click();
    await vi.waitFor(() => expect(gate.state()).toBe("ready"));
    expect(load).toHaveBeenCalledTimes(2);
  });

  it("remove() detaches the element and stops listening", () => {
    const { host, control, ensure, el } = setup();
    control.remove();
    expect(host.querySelector(".brain-open")).toBeNull();
    el.click();
    expect(ensure).not.toHaveBeenCalled();
  });
});
