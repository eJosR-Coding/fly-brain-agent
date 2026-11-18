import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import { OBJECT_COLORS } from "../src/bodyObjects";
import { createSandbox } from "../src/bodySandbox";

const theme = { highlightHex: "#ffffff" } as never;
const make = () => createSandbox(theme, -0.4, 50);
const named = (g: THREE.Object3D, n: string) => g.getObjectByName(n) as THREE.Mesh & THREE.LineSegments;

describe("createSandbox (glass cube, floor grid, objects layer)", () => {
  it("builds edges, very faint faces, a floor grid and an objects group, hidden until shown", () => {
    const s = make();
    for (const n of ["cube-edges", "cube-faces", "cube-floor", "objects"]) expect(named(s.group, n)).toBeTruthy();
    expect(s.group.visible).toBe(false);
    s.setVisible(true);
    expect(s.group.visible).toBe(true);
  });
  it("draws the faces as glass: faint, no depth writes (the fly inside stays visible), after the fly's own lines", () => {
    const m = named(make().group, "cube-faces").material as THREE.MeshBasicMaterial;
    expect(m.transparent).toBe(true);
    expect(m.depthWrite).toBe(false);
    expect(m.depthTest).toBe(true);
    expect(m.opacity).toBeGreaterThan(0);
    expect(m.opacity).toBeLessThanOrEqual(0.08);
    expect(named(make().group, "cube-faces").renderOrder).toBeGreaterThan(0);
  });
  it("draws the edges in the theme line colour, stronger than the faces, depth tested", () => {
    const s = make();
    const e = named(s.group, "cube-edges").material as THREE.LineBasicMaterial;
    const f = named(s.group, "cube-faces").material as THREE.MeshBasicMaterial;
    expect(e.color.getHexString()).toBe("ffffff");
    expect(e.opacity).toBeGreaterThan(f.opacity);
    expect(e.depthTest).toBe(true);
    expect(e.depthWrite).toBe(false);
  });
  it("sizes the cube from the interior and puts its floor at the body's floor", () => {
    const s = make();
    expect(s.bounds.min).toEqual([-25, -0.4, -25]);
    expect(s.bounds.max).toEqual([25, 49.6, 25]);
    const edges = named(s.group, "cube-edges").geometry.getAttribute("position");
    expect(edges.count).toBe(24);
    const floor = named(s.group, "cube-floor");
    expect(floor.position.y).toBeCloseTo(-0.4, 9);
  });
  it("renders objects as coloured cubes at floor-relative positions", () => {
    const s = make();
    const r = s.setObjects([{ id: "a", kind: "sugar", pos_mm: [5, 1, -3], size_mm: 2 }, { id: "b", kind: "coffee", pos_mm: [-5, 1.5, 4], size_mm: 3 }]);
    expect(r.errors).toEqual([]);
    const objs = named(s.group, "objects");
    expect(objs.children).toHaveLength(2);
    const a = objs.getObjectByName("object:a") as THREE.Mesh;
    expect(a.position.toArray()).toEqual([5, -0.4 + 1, -3]);
    expect(a.scale.x).toBe(2);
    expect((a.material as THREE.MeshBasicMaterial).color.getHex()).toBe(OBJECT_COLORS.sugar);
    expect(((objs.getObjectByName("object:b") as THREE.Mesh).material as THREE.MeshBasicMaterial).color.getHex()).toBe(OBJECT_COLORS.coffee);
  });
  it("replaces the layer on every call and clears it with an empty list", () => {
    const s = make();
    s.setObjects([{ id: "a", kind: "sugar", pos_mm: [0, 1, 0], size_mm: 2 }]);
    s.setObjects([{ id: "b", kind: "sugar", pos_mm: [1, 1, 0], size_mm: 2 }]);
    const objs = named(s.group, "objects");
    expect(objs.children.map((c) => c.name)).toEqual(["object:b"]);
    s.setObjects([]);
    expect(objs.children).toHaveLength(0);
  });
  it("shows only the valid objects and reports the rest", () => {
    const s = make();
    const r = s.setObjects([{ id: "far", kind: "sugar", pos_mm: [99, 1, 0], size_mm: 2 }, { id: "ok", kind: "coffee", pos_mm: [0, 1, 0], size_mm: 2 }]);
    expect(r.errors).toHaveLength(1);
    expect(named(s.group, "objects").children.map((c) => c.name)).toEqual(["object:ok"]);
  });
  it("shares one geometry between all objects and releases everything on dispose", () => {
    const geo = vi.spyOn(THREE.BufferGeometry.prototype, "dispose");
    const mat = vi.spyOn(THREE.Material.prototype, "dispose");
    const s = make();
    s.setObjects(Array.from({ length: 5 }, (_, i) => ({ id: `s${i}`, kind: "sugar" as const, pos_mm: [i, 1, 0] as [number, number, number], size_mm: 2 })));
    const geometries = new Set<THREE.BufferGeometry>();
    s.group.traverse((o) => { const g = (o as THREE.Mesh).geometry; if (g) geometries.add(g); });
    expect(geometries.size).toBe(5); // cube edges, faces, floor grid, the unit box and its edges: nothing per object
    geo.mockClear(); mat.mockClear();
    s.dispose();
    expect(geo).toHaveBeenCalledTimes(geometries.size);
    expect(mat.mock.calls.length).toBeGreaterThanOrEqual(5);
    geo.mockRestore(); mat.mockRestore();
  });
});
