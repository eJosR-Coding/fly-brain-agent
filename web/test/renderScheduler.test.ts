import { describe, expect, it } from "vitest";
import { createRenderScheduler } from "../src/renderScheduler";

/** A manual frame clock: schedule() queues, flush() runs one frame. */
function clock() {
  let queue: { id: number; cb: (t: number) => void }[] = [];
  let next = 1;
  return {
    schedule: (cb: (t: number) => void) => { const id = next++; queue.push({ id, cb }); return id; },
    cancel: (id: number) => { queue = queue.filter((q) => q.id !== id); },
    flush(t = 0) { const q = queue; queue = []; q.forEach((x) => x.cb(t)); },
    get pending() { return queue.length; },
  };
}

describe("createRenderScheduler", () => {
  it("draws nothing until something invalidates", () => {
    const c = clock();
    let draws = 0;
    createRenderScheduler({ render: () => { draws++; return false; }, ...c });
    c.flush();
    expect(draws).toBe(0);
    expect(c.pending).toBe(0);
  });

  it("many invalidations in one tick cost one frame", () => {
    const c = clock();
    let draws = 0;
    const s = createRenderScheduler({ render: () => { draws++; return false; }, ...c });
    s.invalidate(); s.invalidate(); s.invalidate();
    expect(c.pending).toBe(1);
    c.flush();
    expect(draws).toBe(1);
    expect(c.pending).toBe(0); // idle again: the loop stopped
  });

  it("keeps the loop running while render reports it is still animating, then stops", () => {
    const c = clock();
    let left = 3;
    let draws = 0;
    const s = createRenderScheduler({ render: () => { draws++; return --left > 0; }, ...c });
    s.invalidate();
    for (let i = 0; i < 10; i++) c.flush();
    expect(draws).toBe(3);
    expect(c.pending).toBe(0);
  });

  it("an invalidation while rendering does not double-schedule", () => {
    const c = clock();
    let s!: ReturnType<typeof createRenderScheduler>;
    s = createRenderScheduler({ render: () => { s.invalidate(); return false; }, ...c });
    s.invalidate();
    c.flush();
    expect(c.pending).toBe(0);
  });

  it("can be woken again after going idle", () => {
    const c = clock();
    let draws = 0;
    const s = createRenderScheduler({ render: () => { draws++; return false; }, ...c });
    s.invalidate(); c.flush();
    s.invalidate(); c.flush();
    expect(draws).toBe(2);
  });

  it("passes the frame timestamp to render", () => {
    const c = clock();
    const seen: number[] = [];
    const s = createRenderScheduler({ render: (t) => { seen.push(t); return false; }, ...c });
    s.invalidate(); c.flush(1234);
    expect(seen).toEqual([1234]);
  });

  it("dispose cancels a pending frame and ignores later invalidations", () => {
    const c = clock();
    let draws = 0;
    const s = createRenderScheduler({ render: () => { draws++; return true; }, ...c });
    s.invalidate();
    s.dispose();
    s.invalidate();
    c.flush();
    expect(draws).toBe(0);
    expect(c.pending).toBe(0);
  });

  it("notifies frame listeners after each draw, and unsubscribe works", () => {
    const c = clock();
    let n = 0;
    const s = createRenderScheduler({ render: () => false, ...c });
    const off = s.onFrame(() => n++);
    s.invalidate(); c.flush();
    off();
    s.invalidate(); c.flush();
    expect(n).toBe(1);
  });
});
