/** The glass-cube sandbox as three objects: thin edges in the page's line style, very faint faces, a floor grid inside and
 *  the objects layer. Built once and toggled; nothing here reads the DOM or a renderer.
 *
 *  Glass, cheaply: the faces are one faint transparent mesh with depthTest on and depthWrite off and a render order after
 *  the fly's own lines. The fly's depth-only occluder (render order -1) therefore hides the faces and edges behind the
 *  body and the faces in front of it blend a little tint over it, so the fly stays visible through the glass. No shaders. */
import * as THREE from "three";
import { CUBE_GRID_CELLS, SANDBOX_INTERIOR_MM, arenaBounds, cubeBounds, type Arena, cubeEdges, cubeFaces, type Bounds } from "./bodyCube";
import { OBJECT_COLORS, OBJECT_KINDS, validateObjects, type ObjectKind, type SandboxObject } from "./bodyObjects";
import { gridSegments } from "./bodyWorld";
import type { Theme } from "./theme";

const EDGE_OPACITY = 0.55;
const FACE_OPACITY = 0.045;
const FLOOR_OPACITY = 0.22;
/** after the fly (lines 0, occluder -1) */
const GLASS_ORDER = 2;

export interface Sandbox {
  group: THREE.Group;
  bounds: Bounds;
  setVisible(on: boolean): void;
  /** Replace the objects layer with the valid objects of `raw`; the rest are reported, never drawn. */
  setObjects(raw: unknown): { errors: string[] };
  dispose(): void;
}

/** `floorY` is the world height of the floor (the fly's lowest point). */
export function createSandbox(theme: Theme, floorY: number, interiorMm: number = SANDBOX_INTERIOR_MM, arena: Arena | null = null): Sandbox {
  if (arena) interiorMm = arena.interiorMm;
  const bounds = arena ? arenaBounds(floorY, arena) : cubeBounds(floorY, interiorMm);
  const disposables: Array<{ dispose(): void }> = [];
  const own = <T extends { dispose(): void }>(d: T): T => { disposables.push(d); return d; };
  const group = new THREE.Group();
  group.name = "sandbox";
  group.visible = false;

  const edgeGeometry = own(new THREE.BufferGeometry());
  edgeGeometry.setAttribute("position", new THREE.BufferAttribute(cubeEdges(bounds), 3));
  const edgeMaterial = own(new THREE.LineBasicMaterial({ color: theme.highlightHex, transparent: true, opacity: EDGE_OPACITY, depthTest: true, depthWrite: false }));
  const edges = new THREE.LineSegments(edgeGeometry, edgeMaterial);
  edges.name = "cube-edges";
  edges.renderOrder = GLASS_ORDER;
  edges.frustumCulled = false;

  const faces = cubeFaces(bounds);
  const faceGeometry = own(new THREE.BufferGeometry());
  faceGeometry.setAttribute("position", new THREE.BufferAttribute(faces.positions, 3));
  faceGeometry.setIndex(new THREE.BufferAttribute(faces.indices, 1));
  const faceMaterial = own(new THREE.MeshBasicMaterial({
    color: theme.highlightHex, transparent: true, opacity: FACE_OPACITY, side: THREE.DoubleSide, depthTest: true, depthWrite: false,
  }));
  const glass = new THREE.Mesh(faceGeometry, faceMaterial);
  glass.name = "cube-faces";
  glass.renderOrder = GLASS_ORDER;
  glass.frustumCulled = false;

  // the floor grid: whole cells across the whole floor, fixed in the world
  const floorGeometry = own(new THREE.BufferGeometry());
  floorGeometry.setAttribute("position", new THREE.BufferAttribute(gridSegments(CUBE_GRID_CELLS / 2, interiorMm / CUBE_GRID_CELLS), 3));
  const floorMaterial = own(new THREE.LineBasicMaterial({ color: theme.highlightHex, transparent: true, opacity: FLOOR_OPACITY, depthWrite: false }));
  const floor = new THREE.LineSegments(floorGeometry, floorMaterial);
  floor.name = "cube-floor";
  floor.renderOrder = -2;
  floor.frustumCulled = false;
  floor.position.y = floorY;

  // objects: one unit box shared by all, scaled per object
  const objects = new THREE.Group();
  objects.name = "objects";
  const boxGeometry = own(new THREE.BoxGeometry(1, 1, 1));
  let sphereGeometry: THREE.SphereGeometry | null = null; // diameter 1: scaled by size_mm; made on the first threat so it is only owned then
  const boxEdges = own(new THREE.EdgesGeometry(boxGeometry));
  const outline = own(new THREE.LineBasicMaterial({ color: theme.highlightHex, transparent: true, opacity: 0.5, depthWrite: false }));
  const kindMaterial = Object.fromEntries(OBJECT_KINDS.map((k) => [k, own(new THREE.MeshBasicMaterial({ color: OBJECT_COLORS[k] }))])) as Record<ObjectKind, THREE.MeshBasicMaterial>;

  group.add(floor, edges, glass, objects);

  const place = (o: SandboxObject) => {
    const mesh = new THREE.Mesh(o.kind === "threat" ? (sphereGeometry ??= own(new THREE.SphereGeometry(0.5, 16, 12))) : boxGeometry, kindMaterial[o.kind]);
    mesh.name = `object:${o.id}`;
    mesh.position.set(o.pos_mm[0], floorY + o.pos_mm[1], o.pos_mm[2]);
    mesh.scale.setScalar(o.size_mm);
    mesh.frustumCulled = false;
    const lines = new THREE.LineSegments(boxEdges, outline);
    lines.frustumCulled = false;
    if (o.kind !== "threat") mesh.add(lines);
    objects.add(mesh);
  };

  return {
    group,
    bounds,
    setVisible(on) { group.visible = on; },
    setObjects(raw) {
      objects.clear(); // geometries and materials are shared and owned above
      const r = validateObjects(raw, bounds);
      r.objects.forEach(place);
      return { errors: r.errors };
    },
    dispose() {
      objects.clear();
      for (const d of disposables.splice(0)) d.dispose();
    },
  };
}
