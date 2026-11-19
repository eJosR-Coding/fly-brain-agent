/** The frames of a live session (/ws/live): the server's `started` header and `chunk` messages -> the same WalkClip the
 *  recorded clips use, so the leg matrices and the root transform come from the existing clip maths (bodyClip).
 *  Pure: no DOM, no three, no socket. A chunk is validated whole; anything malformed throws and the caller drops it. */
import { parseWalkClip, type WalkClip } from "./bodyClip";

type Vec3 = [number, number, number];

/** One body frame is 5 ms (200 Hz); a 50 ms chunk holds 10 of them. */
export const LIVE_FRAME_MS = 5;

export interface LiveHeader {
  bodies: string[];
  sampleRateHz: number;
  /** the header as the server sent it (format, neutral pose, scales): reused to build each chunk's clip */
  raw: Record<string, unknown>;
}

/** What the status line shows about a chunk; every field is the server's own and null when it did not send one. */
export interface LiveInfo {
  speedLabel: string | null;
  speedFactor: number | null;
  backend: string | null;
  fallback: string | null;
  bodyMode: string | null;
  /** the world rule stopped the fly at a glass wall during this chunk */
  worldBlocked: boolean;
}

export interface LiveChunk {
  seq: number;
  /** simulated time at the END of the chunk */
  tMs: number;
  chunkMs: number;
  clip: WalkClip;
  info: LiveInfo;
}

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const str = (v: unknown): string | null => (typeof v === "string" && v !== "" ? v : null);

/** Validate the header by building a one-frame clip held in the neutral pose: parseWalkClip checks every field. */
export function parseLiveHeader(raw: unknown): LiveHeader {
  if (typeof raw !== "object" || raw === null) throw new Error("live header: not an object");
  const r = raw as Record<string, unknown>;
  const neutral = r.neutral as { pos_um?: unknown; quat?: unknown } | undefined;
  if (!Array.isArray(r.bodies) || !neutral || !Array.isArray(neutral.pos_um) || !Array.isArray(neutral.quat)) {
    throw new Error("live header: needs bodies and a neutral pose");
  }
  const frame = neutral.pos_um.map((p, i) => [...(p as number[]), ...((neutral.quat as number[][])[i] ?? [])]);
  const clip = parseWalkClip({ ...r, n_frames: 1, frames: [frame], loop: false });
  return { bodies: clip.bodies, sampleRateHz: clip.sampleRateHz, raw: r };
}

const asRows = (v: unknown, n: number, len: number, what: string): number[][] => {
  if (!Array.isArray(v) || v.length !== n || !v.every((row) => Array.isArray(row) && row.length === len && row.every(isNum))) {
    throw new Error(`live chunk: ${what} must be ${n} rows of ${len} finite numbers`);
  }
  return v as number[][];
};

/** One chunk message -> its clip. `z0` is the thorax height (um) of the first frame of the session (null before it is
 *  known): the server sends x and y relative to the start but z absolute, and the clip wants all three relative. Returns
 *  the new `z0` to carry to the next chunk. Throws on anything malformed (including frames sent as binary). */
export function parseLiveChunk(raw: unknown, header: LiveHeader, z0: number | null): { chunk: LiveChunk; z0: number } {
  if (typeof raw !== "object" || raw === null) throw new Error("live chunk: not an object");
  const m = raw as Record<string, unknown>;
  if (!Number.isInteger(m.seq) || (m.seq as number) < 0) throw new Error("live chunk: bad seq");
  if (!isNum(m.t_ms) || !isNum(m.chunk_ms) || m.chunk_ms <= 0) throw new Error("live chunk: bad t_ms or chunk_ms");
  const frames = m.frames;
  if (!Array.isArray(frames) || frames.length === 0) throw new Error("live chunk: no frames (binary frames are not requested)");
  const n = frames.length;
  const th = m.thorax as { pos_um?: unknown; yaw_deg?: unknown } | undefined;
  if (typeof th !== "object" || th === null) throw new Error("live chunk: no thorax");
  const pos = asRows(th.pos_um, n, 3, "thorax.pos_um");
  if (!Array.isArray(th.yaw_deg) || th.yaw_deg.length !== n || !th.yaw_deg.every(isNum)) throw new Error("live chunk: thorax.yaw_deg must be one angle per frame");
  const base = z0 ?? pos[0][2];
  // parseWalkClip validates frames (body count, 7 finite numbers each) and normalises the quaternions
  const clip = parseWalkClip({
    ...header.raw, n_frames: n, frames, loop: false, escape: new Array(n).fill(m.escape === true ? 1 : 0),
  });
  const p0 = (a: number[]): Vec3 => [a[1] / 1000, (a[2] - base) / 1000, a[0] / 1000]; // flygym (x fwd, y left, z up) -> viewer, mm
  const world = new Float64Array(n * 3);
  pos.forEach((p, i) => world.set(p0(p), i * 3));
  if (Array.isArray(m.wings) && m.wings.length === n && m.wings.every((r) => Array.isArray(r) && r.length === 6 && r.every(isNum))) {
    clip.wings = Float64Array.from((m.wings as number[][]).flat());
  }
  if (Array.isArray(m.wing_cycle) && m.wing_cycle.length >= 2 && m.wing_cycle.every((r) => Array.isArray(r) && r.length === 6 && r.every(isNum))) {
    clip.wingCycle = Float64Array.from((m.wing_cycle as number[][]).flat());
  }
  clip.world = { pos: world, yawRad: Float64Array.from(th.yaw_deg as number[], (d) => (d * Math.PI) / 180) };
  return {
    z0: base,
    chunk: {
      seq: m.seq as number, tMs: m.t_ms, chunkMs: m.chunk_ms, clip,
      info: {
        speedLabel: str(m.speed_label), speedFactor: isNum(m.speed_factor) ? m.speed_factor : null,
        backend: str(m.backend), fallback: str(m.fallback), bodyMode: str(m.body_mode),
        worldBlocked: (m.world as { blocked?: unknown } | undefined)?.blocked === true,
      },
    },
  };
}
