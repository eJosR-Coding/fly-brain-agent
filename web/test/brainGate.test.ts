import { describe, expect, it, vi } from "vitest";
import { createBrainGate, isActivationKey, isZoomIntent } from "../src/brainGate";

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

describe("createBrainGate", () => {
  it("starts idle and does not load until ensure()", () => {
    const load = vi.fn(async () => 1);
    const g = createBrainGate({ load });
    expect(g.state()).toBe("idle");
    expect(load).not.toHaveBeenCalled();
  });

  it("goes loading -> ready and resolves with the loaded value", async () => {
    const d = deferred<string>();
    const g = createBrainGate({ load: () => d.promise });
    const p = g.ensure();
    expect(g.state()).toBe("loading");
    d.resolve("scene");
    await expect(p).resolves.toBe("scene");
    expect(g.state()).toBe("ready");
  });

  it("concurrent callers share ONE load and later calls reuse the result", async () => {
    const d = deferred<number>();
    const load = vi.fn(() => d.promise);
    const g = createBrainGate({ load });
    const a = g.ensure();
    const b = g.ensure();
    d.resolve(7);
    expect(await a).toBe(7);
    expect(await b).toBe(7);
    expect(await g.ensure()).toBe(7);
    expect(load).toHaveBeenCalledTimes(1);
  });

  it("failure moves to failed, rejects every waiter, and a later ensure() retries", async () => {
    const d = deferred<number>();
    const load = vi.fn().mockReturnValueOnce(d.promise).mockResolvedValueOnce(9);
    const g = createBrainGate({ load });
    const a = g.ensure();
    const b = g.ensure();
    d.reject(new Error("offline"));
    await expect(a).rejects.toThrow("offline");
    await expect(b).rejects.toThrow("offline");
    expect(g.state()).toBe("failed");
    await expect(g.ensure()).resolves.toBe(9);
    expect(g.state()).toBe("ready");
    expect(load).toHaveBeenCalledTimes(2);
  });

  it("treats a synchronous throw of load as a failure", async () => {
    const g = createBrainGate<number>({ load: () => { throw new Error("boom"); } });
    await expect(g.ensure()).rejects.toThrow("boom");
    expect(g.state()).toBe("failed");
  });

  it("notifies subscribers on every transition, until unsubscribed", async () => {
    const g = createBrainGate({ load: async () => 1 });
    const seen: string[] = [];
    const off = g.subscribe((s) => seen.push(s));
    await g.ensure();
    expect(seen).toEqual(["loading", "ready"]);
    off();
    const g2 = createBrainGate({ load: async () => 1 });
    const seen2: string[] = [];
    const off2 = g2.subscribe((s) => seen2.push(s));
    off2();
    await g2.ensure();
    expect(seen2).toEqual([]);
  });

  it("a throwing subscriber does not break the load", async () => {
    const g = createBrainGate({ load: async () => 5 });
    g.subscribe(() => { throw new Error("bad listener"); });
    await expect(g.ensure()).resolves.toBe(5);
  });
});

describe("open-trigger helpers", () => {
  it("isZoomIntent: wheel zoom-in (deltaY < 0) or ctrl pinch, not scroll-out", () => {
    expect(isZoomIntent({ deltaY: -10, ctrlKey: false })).toBe(true);
    expect(isZoomIntent({ deltaY: 10, ctrlKey: true })).toBe(true);
    expect(isZoomIntent({ deltaY: -1, ctrlKey: true })).toBe(true);
    expect(isZoomIntent({ deltaY: 10, ctrlKey: false })).toBe(false);
    expect(isZoomIntent({ deltaY: 0, ctrlKey: false })).toBe(false);
  });
  it("isActivationKey: Enter and Space only", () => {
    expect(isActivationKey("Enter")).toBe(true);
    expect(isActivationKey(" ")).toBe(true);
    expect(isActivationKey("Spacebar")).toBe(true);
    expect(isActivationKey("a")).toBe(false);
    expect(isActivationKey("Tab")).toBe(false);
  });
});
