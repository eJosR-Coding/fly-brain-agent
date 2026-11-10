/** Neurotransmitter layer: one uint8 per neuron (/brain/nt) mapped to grey tiers, plus legend rows. Pure. */




/** Grey tone (0..1) and point-size multiplier per transmitter. Rare, interesting ones are the brightest. */
export const NT_TIERS: Record<string, { tone: number; size: number }> = {
  dopamine: { tone: 1.0, size: 1.9 },
  serotonin: { tone: 0.9, size: 1.9 },
  octopamine: { tone: 0.78, size: 1.7 },
  gaba: { tone: 0.62, size: 1.2 },
  glutamate: { tone: 0.44, size: 1 },
  acetylcholine: { tone: 0.27, size: 1 },
  unknown: { tone: 0.13, size: 0.9 },
};

export const ntTier = (name: string) => NT_TIERS[name] ?? NT_TIERS.unknown;

const MAX_CODE = 255;
function entries(table: Record<string, string>): [number, string][] {
  return Object.entries(table)
    .filter(([k]) => /^\d+$/.test(k) && Number(k) <= MAX_CODE)
    .map(([k, name]) => [Number(k), name]);
}

/** Grey rgb and size multiplier per code (index = code). */
export function ntPalette(table: Record<string, string>): { palette: Float32Array; sizes: Float32Array } {
  const e = entries(table);
  const n = e.reduce((m, [c]) => Math.max(m, c + 1), 0);
  const palette = new Float32Array(n * 3);
  const sizes = new Float32Array(n);
  for (let i = 0; i < n; i++) { palette.fill(NT_TIERS.unknown.tone, i * 3, i * 3 + 3); sizes[i] = NT_TIERS.unknown.size; }
  for (const [code, name] of e) {
    const t = ntTier(name);
    palette.fill(t.tone, code * 3, code * 3 + 3);
    sizes[code] = t.size;
  }
  return { palette, sizes };
}

/** One size multiplier per point; a code past the table gets the last entry. */
export function ntWeights(codes: Uint8Array, sizes: Float32Array): Float32Array {
  const out = new Float32Array(codes.length);
  const last = sizes.length - 1;
  if (last < 0) return out.fill(1);
  for (let i = 0; i < codes.length; i++) out[i] = sizes[codes[i] > last ? last : codes[i]];
  return out;
}

export const greyHex = (tone: number): string => {
  const v = Math.round(Math.min(1, Math.max(0, tone)) * 255).toString(16).padStart(2, "0");
  return `#${v}${v}${v}`;
};

export interface KeyRow { code: number; name: string; count: number; hex: string }

/** Legend rows: how many neurons carry each code (table order, empty codes skipped, codes outside the table ignored). */
export function keyRows(codes: Uint8Array, table: Record<string, string>, swatch: (name: string) => string): KeyRow[] {
  const counts = new Map<number, number>();
  for (let i = 0; i < codes.length; i++) counts.set(codes[i], (counts.get(codes[i]) ?? 0) + 1);
  return entries(table)
    .sort((a, b) => a[0] - b[0])
    .filter(([code]) => counts.has(code))
    .map(([code, name]) => ({ code, name, count: counts.get(code)!, hex: swatch(name) }));
}

export interface Transmitters { codes: Uint8Array; table: Record<string, string> }

export async function loadNt(count: number, table: Record<string, string>, fetchFn: typeof fetch = fetch): Promise<Transmitters> {
  const res = await fetchFn("/api/brain/nt");
  if (!res.ok) throw new Error(`transmitters failed: HTTP ${res.status}`);
  const codes = new Uint8Array(await res.arrayBuffer());
  if (codes.length !== count) throw new Error(`transmitters length ${codes.length} bytes does not match ${count} points`);
  return { codes, table };
}

