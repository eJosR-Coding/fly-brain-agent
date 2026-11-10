/** Pure helper for the highlight layer: which points to draw, as an index buffer. */

/**
 * Write the unique, in-range point indices of a selection into `target` (ascending) and return
 * how many were written. `target.length` is the total point count. The caller draws
 * `[0, n)` of an indexed THREE.Points that shares the cloud's position buffer, so a highlight
 * of up to every point costs one linear pass and no position copy.
 */
export function writeSelectionIndex(target: Uint32Array, indices: ArrayLike<number>): number {
  const count = target.length;
  const seen = new Uint8Array(count);
  for (let k = 0; k < indices.length; k++) {
    const i = indices[k];
    if (Number.isInteger(i) && i >= 0 && i < count) seen[i] = 1;
  }
  let n = 0;
  for (let i = 0; i < count; i++) if (seen[i]) target[n++] = i;
  return n;
}
