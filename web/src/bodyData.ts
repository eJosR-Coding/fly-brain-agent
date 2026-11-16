/** Fly body asset: line segments of the NeuroMechFly v2 micro-CT mesh, packaged as /fly-body.json + /fly-body.bin.
 *  Pure decoding (no DOM, no three). Bin: per segment two endpoints, each uint16 x,y,z little-endian (12 bytes),
 *  quantized per axis over meta.bounds (um = min + q / 65535 * (max - min)). */

export type Vec3 = [number, number, number];

export interface BodyPartMeta {
  name: string;
  group: string;
  side: string;
  segment_start: number;
  segment_count: number;
}

export interface BodyMeta {
  /** [min, max] in micrometres, fly frame: x anterior(+), y left(+), z dorsal(+) */
  bounds: [Vec3, Vec3];
  head_center_um: Vec3;
  parts: BodyPartMeta[];
}

export interface BodyPart {
  name: string;
  group: string;
  side: string;
  start: number;
  count: number;
}

export interface ParsedBody {
  /** xyz per endpoint, 2 endpoints per segment, millimetres, Three.js axes */
  positions: Float32Array;
  segmentCount: number;
  bounds: { min: Vec3; max: Vec3 };
  center: Vec3;
  headCenter: Vec3;
  parts: BodyPart[];
}

const BYTES_PER_SEGMENT = 12;
const UM_TO_MM = 1 / 1000;

/** Fly frame (x anterior, y left, z dorsal) -> Three.js axes (X right of the viewer, Y up, Z toward the viewer):
 *  three = (y_fly, z_fly, x_fly). It is a cyclic permutation, so handedness is kept (x cross y = z maps to
 *  Z cross X = Y), and a fly facing the camera has its left on the viewer's right. */
const toThree = (x: number, y: number, z: number): Vec3 => [y, z, x];

export function parseBody(buffer: ArrayBuffer, meta: BodyMeta): ParsedBody {
  const covered = meta.parts.reduce((n, p) => n + p.segment_count, 0);
  if (buffer.byteLength !== covered * BYTES_PER_SEGMENT) {
    throw new Error(`fly body is ${buffer.byteLength} bytes, its parts describe ${covered} segments (needs ${covered * BYTES_PER_SEGMENT})`);
  }
  const segmentCount = covered;
  let next = 0;
  for (const p of meta.parts) {
    if (p.segment_start !== next) throw new Error(`fly body parts are not contiguous at ${p.name} (starts at ${p.segment_start}, expected ${next})`);
    next += p.segment_count;
  }
  const [lo, hi] = meta.bounds;
  const view = new DataView(buffer);
  const positions = new Float32Array(segmentCount * 6);
  for (let i = 0; i < segmentCount * 2; i++) {
    const q = (a: number) => view.getUint16((i * 3 + a) * 2, true);
    const fx = (lo[0] + (q(0) / 65535) * (hi[0] - lo[0])) * UM_TO_MM;
    const fy = (lo[1] + (q(1) / 65535) * (hi[1] - lo[1])) * UM_TO_MM;
    const fz = (lo[2] + (q(2) / 65535) * (hi[2] - lo[2])) * UM_TO_MM;
    positions.set(toThree(fx, fy, fz), i * 3);
  }
  const min = toThree(lo[0] * UM_TO_MM, lo[1] * UM_TO_MM, lo[2] * UM_TO_MM);
  const max = toThree(hi[0] * UM_TO_MM, hi[1] * UM_TO_MM, hi[2] * UM_TO_MM);
  const [hx, hy, hz] = meta.head_center_um;
  return {
    positions,
    segmentCount,
    bounds: { min, max },
    center: [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2],
    headCenter: toThree(hx * UM_TO_MM, hy * UM_TO_MM, hz * UM_TO_MM),
    parts: meta.parts.map((p) => ({ name: p.name, group: p.group, side: p.side, start: p.segment_start, count: p.segment_count })),
  };
}

/** The body is a static file in web/public, served from the site root (not under /api). */
export async function loadBody(fetchFn: typeof fetch = fetch, base = "/"): Promise<ParsedBody> {
  const get = async (name: string) => {
    const res = await fetchFn(`${base}${name}`);
    if (!res.ok) throw new Error(`could not load ${name}: HTTP ${res.status}`);
    return res;
  };
  const [metaRes, binRes] = await Promise.all([get("fly-body.json"), get("fly-body.bin")]);
  return parseBody(await binRes.arrayBuffer(), (await metaRes.json()) as BodyMeta);
}

/** Camera distance that fits the body's bounding sphere in both the vertical and horizontal view, with ~10% margin. */
export function frameDistance(bounds: { min: Vec3; max: Vec3 }, fovDeg: number, aspect: number): number {
  const radius = Math.hypot(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], bounds.max[2] - bounds.min[2]) / 2;
  const halfV = (fovDeg * Math.PI) / 360;
  const halfH = Math.atan(Math.tan(halfV) * Math.max(aspect, 0.01));
  return (radius * 1.1) / Math.sin(Math.min(halfV, halfH));
}
