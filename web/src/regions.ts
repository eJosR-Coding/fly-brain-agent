/** Region context: one uint8 per neuron (/brain/regions) turned into per-point colors, plus /brain/landmarks. */
import type { Vec3 } from "./brainData";
import type { Landmarks, Side } from "./anatomy";

export type Rgb = [number, number, number];
const MAX_CODE = 255;

export function hexToRgb(hex: string): Rgb {
  const m = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return [0.5, 0.5, 0.5];
  let h = m[1];
  if (h.length === 3) h = [...h].map((c) => c + c).join("");
  const n = parseInt(h, 16);
  return [(n >> 16) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

/** rgb per region code (index = code), from the meta table {"0": "central", ...} and a name -> hex tone map. */
export function regionPalette(table: Record<string, string>, tones: Record<string, string>, fallbackHex: string): Float32Array {
  const entries: [number, string][] = [];
  for (const [k, name] of Object.entries(table)) {
    const code = Number(k);
    if (Number.isInteger(code) && code >= 0 && code <= MAX_CODE && /^\d+$/.test(k)) entries.push([code, name]);
  }
  const size = entries.reduce((m, [c]) => Math.max(m, c + 1), 0);
  const palette = new Float32Array(size * 3);
  const fb = hexToRgb(fallbackHex);
  for (let i = 0; i < size; i++) palette.set(fb, i * 3);
  for (const [code, name] of entries) palette.set(name in tones ? hexToRgb(tones[name]) : fb, code * 3);
  return palette;
}

/** One rgb per point. A code outside the palette gets the last entry (never undefined/NaN). */
export function buildRegionColors(codes: Uint8Array, palette: Float32Array): Float32Array {
  const out = new Float32Array(codes.length * 3);
  const last = palette.length / 3 - 1;
  if (last < 0) return out.fill(0.5);
  for (let i = 0; i < codes.length; i++) {
    const c = codes[i] > last ? last : codes[i];
    out[i * 3] = palette[c * 3];
    out[i * 3 + 1] = palette[c * 3 + 1];
    out[i * 3 + 2] = palette[c * 3 + 2];
  }
  return out;
}

export interface Regions {
  codes: Uint8Array;
  table: Record<string, string>;
}

export async function loadRegions(count: number, table: Record<string, string>, fetchFn: typeof fetch = fetch): Promise<Regions> {
  const res = await fetchFn("/api/brain/regions");
  if (!res.ok) throw new Error(`regions failed: HTTP ${res.status}`);
  const codes = new Uint8Array(await res.arrayBuffer());
  if (codes.length !== count) throw new Error(`regions length ${codes.length} bytes does not match ${count} points`);
  return { codes, table };
}

const finite3 = (v: unknown): v is Vec3 => Array.isArray(v) && v.length === 3 && v.every((n) => typeof n === "number" && Number.isFinite(n));

function parseSide(s: any): Side | null {
  if (!s || !finite3(s.lobe_center) || typeof s.radius !== "number" || !Number.isFinite(s.radius)) return null;
  const dir = s.lamina_direction;
  if (dir !== null && !finite3(dir)) return null;
  return { lobe_center: s.lobe_center, radius: s.radius, lamina_direction: dir };
}

export async function loadLandmarks(fetchFn: typeof fetch = fetch): Promise<Landmarks> {
  const res = await fetchFn("/api/brain/landmarks");
  if (!res.ok) throw new Error(`landmarks failed: HTTP ${res.status}`);
  const body: any = await res.json();
  if (!body || !Array.isArray(body.landmarks) || typeof body.sides !== "object" || body.sides === null) {
    throw new Error("landmarks: bad payload");
  }
  const landmarks = body.landmarks.filter(
    (m: any) => m && typeof m.name === "string" && [m.x, m.y, m.z, m.count].every((n) => typeof n === "number" && Number.isFinite(n)),
  );
  const sides: Landmarks["sides"] = {};
  for (const key of ["left", "right"] as const) {
    const s = parseSide(body.sides[key]);
    if (s) sides[key] = s;
  }
  return { landmarks, sides };
}
