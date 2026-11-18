/** The objects layer of the sandbox: validated data in, nothing else (no physics, no contact; those come later).
 *  An object is a small cube of sugar or coffee. `pos_mm` is its CENTRE in the viewer world frame (X left, Z forward,
 *  mm, the fly starts at the origin) with Y the height above the floor; `size_mm` is its edge. */
import { containsBox, type Bounds } from "./bodyCube";

export type ObjectKind = "sugar" | "coffee";
export interface SandboxObject { id: string; kind: ObjectKind; pos_mm: [number, number, number]; size_mm: number }

export const OBJECT_KINDS: readonly ObjectKind[] = ["sugar", "coffee"];
export const MAX_OBJECTS = 64;
/** sugar white-ish, coffee dark brown */
export const OBJECT_COLORS: Record<ObjectKind, number> = { sugar: 0xf2efe6, coffee: 0x4a2c1a };

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** Keep the valid objects (copied) and say why each other one was dropped. `bounds` are the cube's, with its floor at
 *  bounds.min[1]: an object must lie fully inside. */
export function validateObjects(raw: unknown, bounds: Bounds): { objects: SandboxObject[]; errors: string[] } {
  if (!Array.isArray(raw)) return { objects: [], errors: ["objects must be a list"] };
  const objects: SandboxObject[] = [];
  const errors: string[] = [];
  const seen = new Set<string>();
  for (const [i, o] of raw.entries()) {
    if (objects.length >= MAX_OBJECTS) { errors.push(`at most ${MAX_OBJECTS} objects are shown; the rest were dropped`); break; }
    if (typeof o !== "object" || o === null) { errors.push(`object ${i}: not an object`); continue; }
    const r = o as Record<string, unknown>;
    if (typeof r.id !== "string" || r.id === "") { errors.push(`object ${i}: id must be a non-empty string`); continue; }
    const tag = `object "${r.id}"`;
    if (seen.has(r.id)) { errors.push(`${tag}: duplicate id`); continue; }
    if (!OBJECT_KINDS.includes(r.kind as ObjectKind)) { errors.push(`${tag}: kind must be one of ${OBJECT_KINDS.join(", ")}`); continue; }
    if (!Array.isArray(r.pos_mm) || r.pos_mm.length !== 3 || !r.pos_mm.every(isNum)) { errors.push(`${tag}: pos_mm must be three finite numbers`); continue; }
    if (!isNum(r.size_mm) || r.size_mm <= 0) { errors.push(`${tag}: size_mm must be a positive number`); continue; }
    const pos = r.pos_mm as [number, number, number];
    // Y is the height above the floor; the bounds are in world height
    if (!containsBox(bounds, [pos[0], bounds.min[1] + pos[1], pos[2]], r.size_mm)) { errors.push(`${tag}: outside the cube`); continue; }
    seen.add(r.id);
    objects.push({ id: r.id, kind: r.kind as ObjectKind, pos_mm: [pos[0], pos[1], pos[2]], size_mm: r.size_mm });
  }
  return { objects, errors };
}
