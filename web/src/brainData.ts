/** Brain data contract: /brain/meta + /brain/points (little-endian float32 xyz). */

export interface BrainMeta {
  count: number;
  bytes: number;
  dtype: string;
  endianness: string;
  layout: string;
  order: string;
  extent_um: [number, number, number];
  /** region code -> name (e.g. {"0": "central"}); absent on older backends */
  regions?: Record<string, string>;
  /** neurotransmitter code -> name (e.g. {"3": "dopamine"}); absent on older backends */
  nt?: Record<string, string>;
}

export type Vec3 = [number, number, number];

export interface Brain {
  meta: BrainMeta;
  positions: Float32Array;
  center: Vec3;
  size: Vec3;
}

export function computeBounds(positions: Float32Array): { min: Vec3; max: Vec3 } {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      const v = positions[i + a];
      if (v < min[a]) min[a] = v;
      if (v > max[a]) max[a] = v;
    }
  }
  return { min, max };
}

export function decodeIndices(buffer: ArrayBuffer): Uint32Array {
  if (buffer.byteLength % 4 !== 0) {
    throw new Error(`index buffer length ${buffer.byteLength} is not a multiple of 4`);
  }
  // wire format is little-endian; read through DataView so it is right on any host
  const view = new DataView(buffer);
  const out = new Uint32Array(buffer.byteLength / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getUint32(i * 4, true);
  return out;
}

async function ok(res: Response, what: string): Promise<Response> {
  if (!res.ok) throw new Error(`${what} failed: HTTP ${res.status}`);
  return res;
}

export async function loadBrain(fetchFn: typeof fetch = fetch, base = "/api"): Promise<Brain> {
  const meta = (await (await ok(await fetchFn(`${base}/brain/meta`), "brain meta")).json()) as BrainMeta;
  const buf = await (await ok(await fetchFn(`${base}/brain/points`), "brain points")).arrayBuffer();
  if (buf.byteLength !== meta.bytes) {
    throw new Error(`brain points are ${buf.byteLength} bytes, meta says ${meta.bytes}`);
  }
  if (buf.byteLength % 12 !== 0) {
    throw new Error(`brain points are ${buf.byteLength} bytes, not a multiple of 12 (3 float32 per neuron)`);
  }
  if (buf.byteLength !== meta.count * 12) {
    throw new Error(`meta says ${meta.count} points but the data holds ${buf.byteLength / 12} (${meta.count} points need ${meta.count * 12} bytes)`);
  }
  const positions = new Float32Array(buf);
  const { min, max } = computeBounds(positions);
  const center: Vec3 = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  const size: Vec3 = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  return { meta, positions, center, size };
}

export async function fetchIndices(url: string, fetchFn: typeof fetch = fetch, base = "/api"): Promise<Uint32Array> {
  return decodeIndices(await (await ok(await fetchFn(`${base}${url}`), "indices")).arrayBuffer());
}
