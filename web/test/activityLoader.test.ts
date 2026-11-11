import { describe, expect, it } from "vitest";
import { Latest } from "../src/latest";
import { loadLatestActivity } from "../src/activityLoader";

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

describe("loadLatestActivity", () => {
  it("shows only the newest requested recording even when an older load finishes last", async () => {
    const latest = new Latest();
    const slow = deferred<string>();
    const fast = deferred<string>();
    const shown: string[] = [];
    const first = loadLatestActivity(latest, () => slow.promise, (r) => { shown.push(r); });
    const second = loadLatestActivity(latest, () => fast.promise, (r) => { shown.push(r); });
    fast.resolve("new");
    slow.resolve("old");
    expect(await second).toBe(true);
    expect(await first).toBe(false);
    expect(shown).toEqual(["new"]);
  });

  it("rethrows the failure of the current load only", async () => {
    const latest = new Latest();
    const bad = deferred<string>();
    const p = loadLatestActivity(latest, () => bad.promise, () => undefined);
    bad.reject(new Error("boom"));
    await expect(p).rejects.toThrow("boom");
    const stale = deferred<string>();
    const old = loadLatestActivity(latest, () => stale.promise, () => undefined);
    await loadLatestActivity(latest, async () => "x", () => undefined);
    stale.reject(new Error("late"));
    expect(await old).toBe(false);
  });
});
