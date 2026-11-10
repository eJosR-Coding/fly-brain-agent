import { describe, expect, it } from "vitest";
import { Latest, runLatest } from "../src/latest";

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

describe("runLatest", () => {
  it("ignores an older slower response once a newer request started", async () => {
    const latest = new Latest();
    const slow = deferred<string>();
    const fast = deferred<string>();
    const applied: string[] = [];
    const first = runLatest(latest, () => slow.promise, (r) => applied.push(r));
    const second = runLatest(latest, () => fast.promise, (r) => applied.push(r));
    fast.resolve("new");
    slow.resolve("old");
    expect(await first).toBe(false);
    expect(await second).toBe(true);
    expect(applied).toEqual(["new"]);
  });

  it("applies a lone request", async () => {
    const latest = new Latest();
    const applied: number[] = [];
    expect(await runLatest(latest, async () => 1, (r) => applied.push(r))).toBe(true);
    expect(applied).toEqual([1]);
  });

  it("a stale failure is swallowed, a current failure is thrown", async () => {
    const latest = new Latest();
    const a = deferred<number>();
    const stale = runLatest(latest, () => a.promise, () => {});
    const current = runLatest(latest, () => Promise.reject(new Error("boom")), () => {});
    a.reject(new Error("old boom"));
    expect(await stale).toBe(false);
    await expect(current).rejects.toThrow("boom");
  });

  it("invalidate() (the clear chip) cancels what is in flight", async () => {
    const latest = new Latest();
    const d = deferred<number>();
    const applied: number[] = [];
    const run = runLatest(latest, () => d.promise, (r) => applied.push(r));
    latest.invalidate();
    d.resolve(7);
    expect(await run).toBe(false);
    expect(applied).toEqual([]);
  });
});
