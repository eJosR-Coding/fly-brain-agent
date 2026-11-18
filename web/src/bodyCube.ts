/** Pure geometry of the glass-cube sandbox the fly lives in (viewer axes: X = fly left, Y = up, Z = fly forward, mm).
 *  No DOM, no three. The cube is centred on the fly's start in X and Z and stands on the body's floor. Its bounds are
 *  only drawn today; the live backend will enforce the walls later and take the same constant. */

type Vec3 = [number, number, number];
export interface Bounds { min: Vec3; max: Vec3 }

/** Interior edge of the cube, mm: about 18 body lengths (the fly is ~2.8 mm long), room to walk some seconds without
 *  hitting a wall, small enough that the whole cube still reads on screen. */
export const SANDBOX_INTERIOR_MM = 50;
/** The floor grid has this many cells along each side (5 mm cells in the default cube). */
export const CUBE_GRID_CELLS = 10;
/** How much of the world the follow camera frames at its default distance, mm across the limiting screen side: the fly
 *  (~3 mm) stays readable and the nearby walls come into view; the user can zoom out to the whole cube. */
export const FOLLOW_SPAN_MM = 24;

export function cubeBounds(floorY: number, interiorMm: number = SANDBOX_INTERIOR_MM): Bounds {
  const h = interiorMm / 2;
  return { min: [-h, floorY, -h], max: [h, floorY + interiorMm, h] };
}

const corner = (b: Bounds, i: number): Vec3 => [i & 1 ? b.max[0] : b.min[0], i & 2 ? b.max[1] : b.min[1], i & 4 ? b.max[2] : b.min[2]];

/** The 12 edges as xyz pairs (24 vertices). */
export function cubeEdges(b: Bounds): Float32Array {
  const out: number[] = [];
  for (let i = 0; i < 8; i++) {
    for (const bit of [1, 2, 4]) {
      if (i & bit) continue;
      out.push(...corner(b, i), ...corner(b, i | bit));
    }
  }
  return Float32Array.from(out);
}

/** The six faces as 24 vertices (4 per face, not shared) and 12 triangles. */
export function cubeFaces(b: Bounds): { positions: Float32Array; indices: Uint16Array } {
  const pos: number[] = [];
  const idx: number[] = [];
  for (let axis = 0; axis < 3; axis++) {
    const u = (axis + 1) % 3, v = (axis + 2) % 3;
    for (const side of [b.min, b.max]) {
      const base = pos.length / 3;
      for (const [a, c] of [[0, 0], [1, 0], [1, 1], [0, 1]]) {
        const p: Vec3 = [0, 0, 0];
        p[axis] = side[axis];
        p[u] = a ? b.max[u] : b.min[u];
        p[v] = c ? b.max[v] : b.min[v];
        pos.push(...p);
      }
      idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
  }
  return { positions: Float32Array.from(pos), indices: Uint16Array.from(idx) };
}

/** True when the axis-aligned box of edge `size` centred at `centre` lies fully inside `b` (Y is world height). */
export function containsBox(b: Bounds, centre: ArrayLike<number>, size: number): boolean {
  const h = size / 2;
  return [0, 1, 2].every((k) => centre[k] - h >= b.min[k] - 1e-9 && centre[k] + h <= b.max[k] + 1e-9);
}

const halfAngles = (fovDeg: number, aspect: number): number => {
  const v = (fovDeg * Math.PI) / 360;
  return Math.min(v, Math.atan(Math.tan(v) * aspect)); // the narrower of the vertical and horizontal half angles
};

/** Camera distance at which `spanMm` just fits the limiting side of a perspective view (fov = vertical, degrees). */
export function spanFitDistance(spanMm: number, fovDeg: number, aspect: number): number {
  return spanMm / 2 / Math.tan(halfAngles(fovDeg, aspect));
}

/** Camera distance at which the whole cube (its bounding sphere) is in view, from any direction. */
export function cubeFitDistance(b: Bounds, fovDeg: number, aspect: number): number {
  const r = Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]) / 2;
  return r / Math.sin(halfAngles(fovDeg, aspect));
}
