/** Pure logic for the simulated-activity wave: decode frames, brightness per bin, playback state. */

/** Sparse recording: only neurons that spiked. `counts` is row-major (nRows x nBins), spike counts per bin. */
export interface Activity {
  rows: Uint32Array;
  counts: Uint8Array;
  nRows: number;
  nBins: number;
  binMs: number;
  durationMs: number;
  /** largest single count, at least 1 (brightness 1 = this many spikes in one bin) */
  peak: number;
  /** neurons above 1 Hz mean rate (as the stimulate tool reports), when the backend sent it; the rows are every neuron that spiked at all */
  activeCount: number | null;
}

/** Decode `GET /simulations/{id}/frames`: nRows uint32 LE row indices, then nRows * nBins uint8 counts. */
export function decodeFrames(buffer: ArrayBuffer, nRows: number, nBins: number, binMs: number): Activity {
  if (!Number.isInteger(nRows) || nRows < 0) throw new Error(`n_rows must be a non-negative integer, got ${nRows}`);
  if (!Number.isInteger(nBins) || nBins < 1) throw new Error(`n_bins must be a positive integer, got ${nBins}`);
  if (!Number.isFinite(binMs) || binMs <= 0) throw new Error(`bin_ms must be positive, got ${binMs}`);
  const expected = nRows * 4 + nRows * nBins;
  if (buffer.byteLength !== expected) {
    throw new Error(`activity frames: expected ${expected} bytes, got ${buffer.byteLength}`);
  }
  const view = new DataView(buffer);
  const rows = new Uint32Array(nRows);
  for (let i = 0; i < nRows; i++) rows[i] = view.getUint32(i * 4, true); // little-endian on any host
  const counts = new Uint8Array(buffer.slice(nRows * 4));
  let peak = 1;
  for (let i = 0; i < counts.length; i++) if (counts[i] > peak) peak = counts[i];
  return { rows, counts, nRows, nBins, binMs, durationMs: nBins * binMs, peak, activeCount: null };
}

const TRAIL_DECAY = 0.6; // each bin after a spike keeps this share of the brightness

const level = (count: number, peak: number) => Math.sqrt(count / peak); // sqrt: faint activity stays visible

/**
 * Brightness (0..1) of every recorded row at `bin`, written into `out[i]` for `rows[i]`.
 * `trail` extra earlier bins linger with a geometric decay, so spikes leave a wave behind them.
 */
export function fillBrightness(a: Activity, bin: number, out: Float32Array, trail = 3): void {
  if (out.length < a.nRows) throw new Error(`output buffer holds ${out.length} rows, the recording has ${a.nRows}`);
  const b = Math.min(Math.max(Math.trunc(Number.isFinite(bin) ? bin : 0), 0), a.nBins - 1);
  const first = Math.max(0, b - Math.max(0, trail));
  for (let i = 0; i < a.nRows; i++) {
    let best = 0;
    const base = i * a.nBins;
    for (let k = first; k <= b; k++) {
      const c = a.counts[base + k];
      if (c) best = Math.max(best, level(c, a.peak) * TRAIL_DECAY ** (b - k));
    }
    out[i] = Math.min(best, 1);
  }
}

/** The whole recording as one still frame (reduced motion): total spikes per row, busiest row = 1. */
export function summedBrightness(a: Activity, out: Float32Array): void {
  if (out.length < a.nRows) throw new Error(`output buffer holds ${out.length} rows, the recording has ${a.nRows}`);
  const totals = new Float64Array(a.nRows);
  let max = 1;
  for (let i = 0; i < a.nRows; i++) {
    let s = 0;
    for (let k = 0; k < a.nBins; k++) s += a.counts[i * a.nBins + k];
    totals[i] = s;
    if (s > max) max = s;
  }
  for (let i = 0; i < a.nRows; i++) out[i] = level(totals[i], max);
}

/** Write values[i] at target[rows[i]] (the per-point brightness attribute); rows outside the cloud are ignored. */
export function scatterBrightness(target: Float32Array, rows: Uint32Array, values: Float32Array): void {
  for (let i = 0; i < rows.length; i++) if (rows[i] < target.length) target[rows[i]] = values[i];
}

// --- playback state (pure: the caller owns the clock) -------------------------------------

export interface Playback {
  timeMs: number;
  playing: boolean;
  loop: boolean;
  durationMs: number;
  /** simulated ms per real ms: below 1 slows the wave down so it can be followed */
  speed: number;
}

export function createPlayback(durationMs: number, opts: { speed?: number; loop?: boolean } = {}): Playback {
  return { timeMs: 0, playing: true, loop: opts.loop ?? false, durationMs, speed: opts.speed ?? 0.4 };
}

/** Advance by `dtMs` of real time. */
export function tick(p: Playback, dtMs: number): Playback {
  if (!p.playing || !Number.isFinite(dtMs) || dtMs <= 0) return p;
  const t = p.timeMs + dtMs * p.speed;
  if (t < p.durationMs) return { ...p, timeMs: t };
  return p.loop ? { ...p, timeMs: t % p.durationMs } : { ...p, timeMs: p.durationMs, playing: false };
}

/** Play/pause; pressing play on a finished (non-looping) run starts over. */
export function togglePlay(p: Playback): Playback {
  if (p.playing) return { ...p, playing: false };
  return { ...p, playing: true, timeMs: p.timeMs >= p.durationMs ? 0 : p.timeMs };
}

/** Jump to a 0..1 fraction of the duration; keeps the play state. */
export function scrub(p: Playback, fraction: number): Playback {
  const f = Number.isFinite(fraction) ? Math.min(Math.max(fraction, 0), 1) : 0;
  return { ...p, timeMs: f * p.durationMs };
}

/** Button text: pause while running, replay on the final frame, play otherwise. */
export function playLabel(p: Playback): "pause" | "play" | "replay" {
  if (p.playing) return "pause";
  return p.timeMs >= p.durationMs ? "replay" : "play";
}

export function binAt(timeMs: number, binMs: number, nBins: number): number {
  return Math.min(Math.max(Math.floor(timeMs / binMs), 0), nBins - 1);
}

// --- text ---------------------------------------------------------------------------------

export function formatTime(ms: number): string {
  return `t = ${Math.max(0, Math.round(ms))} ms`;
}

/** Pending text while POST /simulate runs (like the chat's thinking line). */
export function formatSimulating(durationMs: number, seconds: number): string {
  const s = Number.isFinite(seconds) && seconds > 0 ? Math.floor(seconds) : 0;
  const sim = durationMs / 1000;
  return `simulating ${Number.isInteger(sim) ? sim : sim.toFixed(1)} s of brain activity… · ${s} s`;
}

/** `active` = neurons above 1 Hz (null when unknown), `spiked` = every neuron in the frames. */
export function activityLegend(active: number | null, spiked: number, readout: { label: string; hz: number } | null): string {
  const n = (v: number) => v.toLocaleString("en-US");
  const counts = active === null ? `${n(spiked)} spiked` : `${n(active)} active > 1 Hz · ${n(spiked)} spiked`;
  const base = `activity · ${counts}`;
  return readout ? `${base} · ${readout.label} ${Math.round(readout.hz)} Hz` : base;
}

// --- wire contracts -----------------------------------------------------------------------

export interface SimulationSummary {
  simulationId: string;
  active: number;
  readoutHz: number | null;
  durationMs: number;
}

/** Check the JSON of POST /simulate. Returns the summary, or why it is invalid. */
export function parseSimulation(d: unknown): SimulationSummary | string {
  if (typeof d !== "object" || d === null) return "not an object";
  const r = d as Record<string, unknown>;
  if (typeof r.simulation_id !== "string" || !r.simulation_id) return "simulation_id is missing";
  if (typeof r.active_neurons !== "number" || !Number.isFinite(r.active_neurons)) return "active_neurons is not a number";
  const stim = (r.stimulus ?? {}) as Record<string, unknown>;
  if (typeof stim.duration_ms !== "number" || !(stim.duration_ms > 0)) return "stimulus.duration_ms is missing";
  const ro = r.readout as Record<string, unknown> | undefined;
  const hz = ro && typeof ro.mean_hz === "number" && Number.isFinite(ro.mean_hz) ? ro.mean_hz : null;
  return { simulationId: r.simulation_id, active: r.active_neurons, readoutHz: hz, durationMs: stim.duration_ms };
}

export interface ActivityEvent {
  simulationId: string;
  nBins: number;
  binMs: number;
  /** neurons above 1 Hz, when the backend sends it */
  activeCount: number | null;
}

/** Check the payload of the SSE `activity` event. */
export function parseActivityEvent(d: Record<string, unknown>): ActivityEvent | string {
  if (typeof d.simulation_id !== "string" || !d.simulation_id) return "simulation_id is missing";
  if (typeof d.n_bins !== "number" || !Number.isInteger(d.n_bins) || d.n_bins < 1) return "n_bins is not a positive integer";
  if (typeof d.bin_ms !== "number" || !(d.bin_ms > 0)) return "bin_ms is not positive";
  const ac = typeof d.active_count === "number" && Number.isInteger(d.active_count) && d.active_count >= 0 ? d.active_count : null;
  return { simulationId: d.simulation_id, nBins: d.n_bins, binMs: d.bin_ms, activeCount: ac };
}

async function ok(res: Response, what: string): Promise<Response> {
  if (res.ok) return res;
  let detail = `HTTP ${res.status}`;
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    /* keep the status */
  }
  throw new Error(`${what} failed: ${detail}`);
}

/** Fetch the meta and the binary frames of a simulation and decode them. */
export async function loadActivity(id: string, fetchFn: typeof fetch = fetch, base = "/api"): Promise<Activity> {
  const meta = (await (await ok(await fetchFn(`${base}/simulations/${id}`), "simulation meta")).json()) as Record<string, number>;
  const buf = await (await ok(await fetchFn(`${base}/simulations/${id}/frames`), "simulation frames")).arrayBuffer();
  const a = decodeFrames(buf, meta.n_rows, meta.n_bins, meta.bin_ms);
  const ac = meta.active_count;
  return { ...a, activeCount: typeof ac === "number" && Number.isInteger(ac) && ac >= 0 ? ac : null };
}
