/**
 * Pure mapping from simulated readout rates to joint angles. No DOM, no three.js.
 *
 * Honesty: the body is a static micro-CT mesh posed by us. The INTENSITY of each movement follows a real simulated
 * readout; the movement itself is our illustration. Only MN9 -> proboscis extension has paper backing (Shiu et al. 2024).
 */

export type LegId = `leg_${"lf" | "lm" | "lh"}_${"l" | "r"}`;
/** Three-angle wing pose of flight (degrees): stroke (sweep, + forward), deviation (+ up), rotation about the span. Setting any
 *  of the three for a wing replaces its single "extension" angle; see bodyWorld.flightWingPose for the axis mapping. */
export type WingFlightId = `wing_${"l" | "r"}_${"stroke" | "dev" | "rot"}`;
export type JointId =
  | "proboscis" | "antenna_l" | "antenna_r" | "wing_l" | "wing_r" | "haltere_l" | "haltere_r" | LegId | WingFlightId;

export type Validation = "validated" | "illustrative" | "exploratory";

export interface MappingRow {
  joints: JointId[];
  /** rate (Hz) that gives the full angle; a FIXED constant, see poseFor */
  fullScaleHz: number;
  maxDeg: number;
  kind: "extension" | "oscillation";
  /** only for kind "oscillation" */
  oscillationHz?: number;
  validation: Validation;
  note: string;
}

export const MAPPING: Readonly<Record<string, MappingRow>> = Object.freeze({
  mn9: {
    joints: ["proboscis"], fullScaleHz: 60, maxDeg: 60, kind: "extension", validation: "validated",
    note: "Shiu et al. 2024: MN9 is the proboscis extension motor neuron.",
  },
  abn1: {
    joints: ["antenna_l", "antenna_r"], fullScaleHz: 30, maxDeg: 25, kind: "oscillation", oscillationHz: 6,
    validation: "illustrative",
    note: "aBN1 is the model's antennal grooming readout; the antenna motion itself is illustrative.",
  },
  dnp01: {
    joints: ["wing_l", "wing_r"], fullScaleHz: 100, maxDeg: 70, kind: "extension", validation: "exploratory",
    note: "DNp01 is the giant fiber (escape); looming selectivity was not shown in this model.",
  },
} satisfies Record<string, MappingRow>);

function rowFor(name: string): MappingRow | undefined {
  return Object.prototype.hasOwnProperty.call(MAPPING, name) ? MAPPING[name] : undefined;
}

const clamp01 = (x: number): number => Math.min(1, Math.max(0, x));

/**
 * Joint angles in degrees for the given readout rates at time tMs. Deterministic in its inputs.
 * Intensity = clamp(hz / fullScaleHz, 0, 1) with FIXED constants, never normalized per run, so a low dose looks
 * weaker than a high dose. Non-finite or negative rates count as 0; unknown readouts are ignored. A joint appears in
 * the result only when a known readout drives it; several readouts on one joint combine by the larger absolute angle.
 */
export function poseFor(readoutsHz: Record<string, number>, tMs: number): Partial<Record<JointId, number>> {
  const pose: Partial<Record<JointId, number>> = {};
  for (const name of Object.keys(readoutsHz)) {
    const row = rowFor(name);
    if (!row) continue;
    const hz = readoutsHz[name];
    const intensity = Number.isFinite(hz) && hz > 0 ? clamp01(hz / row.fullScaleHz) : 0;
    let angle = intensity * row.maxDeg;
    if (row.kind === "oscillation") {
      const wave = Number.isFinite(tMs) ? Math.sin(2 * Math.PI * (row.oscillationHz ?? 0) * tMs / 1000) : 0;
      angle *= wave;
    }
    for (const joint of row.joints) {
      const prev = pose[joint];
      pose[joint] = prev === undefined || Math.abs(angle) > Math.abs(prev) ? angle : prev;
    }
  }
  return pose;
}

const SENTENCE = "mapping chosen by us; intensity follows the real readout";

/** One honest on-screen line for the readouts currently driving the body. Unknown names are ignored. */
export function captionFor(activeReadouts: readonly string[]): string {
  const rows = activeReadouts.map(rowFor).filter((r): r is MappingRow => r !== undefined);
  const levels = new Set(rows.map((r) => r.validation));
  const hasIllustrative = levels.has("illustrative");
  const hasExploratory = levels.has("exploratory");
  if (!hasIllustrative && !hasExploratory) {
    if (levels.has("validated")) {
      return "Proboscis extension follows the proboscis motor neuron readout (validated); the movement itself is our illustration, " +
        `${SENTENCE}.`;
    }
    return `Simulated readouts posing a static body: ${SENTENCE}.`;
  }
  const words = [hasIllustrative ? "illustrative" : "", hasExploratory ? "exploratory" : ""].filter(Boolean).join(" and ");
  const lead = levels.has("validated") ? "Proboscis extension is validated; other movements are" : "Movement is";
  return `${lead} ${words}: ${SENTENCE}.`;
}

/**
 * Value of a per-bin series at time tMs. Linear interpolation between bin centres ((i + 0.5) * binMs), clamped to the
 * series ends; smoothBins > 1 averages that many interpolated samples one bin apart, centred on tMs (end values are
 * replicated). Empty series or bad numbers give 0; non-finite or negative bins count as 0.
 */
export function sampleReadout(hz: readonly number[], binMs: number, tMs: number, smoothBins = 1): number {
  const n = hz.length;
  if (n === 0 || !(binMs > 0) || !Number.isFinite(binMs) || !Number.isFinite(tMs)) return 0;
  const at = (t: number): number => {
    const pos = Math.min(n - 1, Math.max(0, t / binMs - 0.5));
    const i = Math.floor(pos);
    const a = hz[i], b = hz[Math.min(n - 1, i + 1)];
    const va = Number.isFinite(a) && a > 0 ? a : 0, vb = Number.isFinite(b) && b > 0 ? b : 0;
    return va + (vb - va) * (pos - i);
  };
  const k = Math.max(1, Math.floor(Number.isFinite(smoothBins) ? smoothBins : 1));
  let sum = 0;
  for (let j = 0; j < k; j++) sum += at(tMs + (j - (k - 1) / 2) * binMs);
  return sum / k;
}
