/** Pure color-buffer helpers (Float32Array, rgb triples in 0..1). */

export type RGB = [number, number, number];

export function baseColors(count: number, rgb: RGB): Float32Array {
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) out.set(rgb, i * 3);
  return out;
}

/** Recolor the given point indices in place; out-of-range indices are ignored. */
export function applyHighlight(colors: Float32Array, indices: ArrayLike<number>, rgb: RGB): Float32Array {
  const n = colors.length / 3;
  for (let k = 0; k < indices.length; k++) {
    const i = indices[k];
    if (i >= 0 && i < n) colors.set(rgb, i * 3);
  }
  return colors;
}

/** Restore the base colors in place. */
export function clearHighlight(colors: Float32Array, base: Float32Array): Float32Array {
  colors.set(base);
  return colors;
}
