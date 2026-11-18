/** Pure parts of the "looming disc (escape)" chip: the validated request and reading the answer. */

/** Validated configuration (scratch validation 2026-11-18): motion front end, right eye, 300 Hz, both polarities. */
export const LOOMING_RUN = {
  input: "motion", side: "right", rate_hz: 300, polarity: "both", trials: 5, duration_ms: 500, steps: 10,
} as const;

export const LOOMING_CONDITIONS = ["looming", "receding", "dimming", "control"] as const;
type Condition = (typeof LOOMING_CONDITIONS)[number];

export interface LoomingRun {
  /** the LOOMING condition's recording: the one the player (and the body) plays */
  simulationId: string;
  durationMs: number;
  /** mean DNp01 rate per condition, Hz */
  dnp01: Record<Condition, number>;
}

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** Check the JSON of POST /experiments/looming. Returns what the chip needs, or why it is invalid. */
export function parseLoomingRun(d: unknown): LoomingRun | string {
  if (typeof d !== "object" || d === null) return "not an object";
  const o = d as Record<string, unknown>;
  const conds = (o.conditions ?? {}) as Record<string, Record<string, unknown> | undefined>;
  const loom = conds.looming;
  if (typeof loom !== "object" || loom === null) return "no looming condition";
  const id = typeof loom.simulation_id === "string" && loom.simulation_id ? loom.simulation_id
    : typeof o.simulation_id === "string" ? o.simulation_id : "";
  if (!id) return "simulation_id is missing";
  const dnp01 = {} as Record<Condition, number>;
  for (const name of LOOMING_CONDITIONS) {
    const hz = conds[name]?.dnp01_hz;
    if (!isNum(hz)) return `${name}.dnp01_hz is not a number`;
    dnp01[name] = hz;
  }
  const stim = (o.stimulus ?? {}) as Record<string, unknown>;
  return { simulationId: id, durationMs: isNum(stim.duration_ms) && stim.duration_ms > 0 ? stim.duration_ms : 500, dnp01 };
}

/** One honest legend line: a model prediction, not an observation. */
export function loomingLegend(r: LoomingRun): string {
  const rates = LOOMING_CONDITIONS.map((c) => `${c} ${Math.round(r.dnp01[c] * 10) / 10} Hz`).join(" · ");
  return `looming disc · model prediction, DNp01 (mean): ${rates}`;
}
