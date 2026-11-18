/** Jitter buffer of a live session. Chunks arrive whenever the computer finishes them (often slower than real time) and
 *  carry 50 ms of motion each; this plays them back on their own time base. Pure: the caller supplies the elapsed wall
 *  time of each draw, nothing here reads a clock.
 *
 *  Rate: 1 (real simulated speed, never faster) while at least `targetMs` of motion is buffered, proportionally slower
 *  below that, 0 (hold the last frame) when dry. A producer slower than real time therefore settles at ITS speed in
 *  smooth slow motion instead of playing 50 ms bursts with freezes in between; the speed the server reports (x0.20 ...)
 *  is what the viewer is showing. Beyond `maxChunks` the oldest chunks are dropped (and counted): that only happens when the
 *  producer outruns playback for a long time. */

export interface PlaybackChunk { chunkMs: number }

export interface PlaybackFrame<T> {
  /** the chunk on screen */
  item: T;
  /** position inside that chunk, seconds (held at its end when dry) */
  tSec: number;
  /** playback speed over simulated time used for this step, 0..1 */
  rate: number;
}

export interface Playback<T extends PlaybackChunk> {
  push(chunk: T): void;
  /** Advance by `dtMs` of wall time. null until the first chunk. */
  advance(dtMs: number): PlaybackFrame<T> | null;
  reset(): void;
  /** simulated ms ahead of the playhead */
  readonly bufferedMs: number;
  /** chunks discarded because the buffer was full */
  readonly dropped: number;
  readonly length: number;
}

export function createPlayback<T extends PlaybackChunk>(opts: { targetMs?: number; maxChunks?: number } = {}): Playback<T> {
  const targetMs = opts.targetMs ?? 100;
  const maxChunks = Math.max(2, opts.maxChunks ?? 6);
  let queue: T[] = [];
  let offMs = 0; // playhead inside queue[0]
  let dropped = 0;
  let lastRate = 0;

  const buffered = () => Math.max(0, queue.reduce((s, c) => s + c.chunkMs, 0) - offMs);
  const settle = () => { // finished chunks leave, but the last one stays on screen
    while (queue.length > 1 && offMs >= queue[0].chunkMs) { offMs -= queue[0].chunkMs; queue.shift(); }
  };
  return {
    push(chunk) {
      queue.push(chunk);
      settle();
      while (queue.length > maxChunks) { queue.shift(); offMs = 0; dropped++; }
    },
    advance(dtMs) {
      if (queue.length === 0) return null;
      if (Number.isFinite(dtMs) && dtMs > 0) {
        lastRate = Math.min(1, buffered() / targetMs);
        offMs += dtMs * lastRate;
        settle();
        offMs = Math.min(offMs, queue[0].chunkMs);
      }
      return { item: queue[0], tSec: offMs / 1000, rate: queue.length === 1 && offMs >= queue[0].chunkMs ? 0 : lastRate };
    },
    reset() { queue = []; offMs = 0; dropped = 0; lastRate = 0; },
    get bufferedMs() { return buffered(); },
    get dropped() { return dropped; },
    get length() { return queue.length; },
  };
}
