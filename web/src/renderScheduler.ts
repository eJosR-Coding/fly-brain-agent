/** Draw-on-demand frame loop: one frame per invalidation, and it keeps going only while render says it is animating. */

export interface RenderSchedulerOptions {
  /** Draw one frame at `now` (ms). Return true while something is still animating (damping, orbit, playback...). */
  render: (now: number) => boolean;
  schedule: (cb: (now: number) => void) => number;
  cancel: (id: number) => void;
}

export interface RenderScheduler {
  /** Something changed: draw a frame soon. Cheap to call many times (they collapse into one frame). */
  invalidate(): void;
  /** Called after every drawn frame (the inspector card uses this to follow the camera). */
  onFrame(cb: () => void): () => void;
  dispose(): void;
}

export function createRenderScheduler({ render, schedule, cancel }: RenderSchedulerOptions): RenderScheduler {
  let id = 0;
  let rendering = false;
  let dead = false;
  const listeners = new Set<() => void>();

  const frame = (now: number) => {
    id = 0;
    if (dead) return;
    rendering = true;
    let again = false;
    try {
      again = render(now);
    } finally {
      rendering = false;
    }
    for (const l of [...listeners]) l();
    if (again && !id && !dead) id = schedule(frame);
  };

  return {
    invalidate() {
      // while rendering, the frame in flight already shows the latest state
      if (dead || id || rendering) return;
      id = schedule(frame);
    },
    onFrame(cb) {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    dispose() {
      dead = true;
      if (id) cancel(id);
      id = 0;
      listeners.clear();
    },
  };
}
