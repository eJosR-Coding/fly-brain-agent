import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { viewFitDistance, type ParsedBody, type Vec3 } from "./bodyData";
import type { JointId } from "./bodyPose";
import type { WorldPose } from "./bodyClip";
import { rotationMatrix, splitByJoint } from "./bodyRig";
import { followStep, gridSegments, gridSnap } from "./bodyWorld";
import { FOLLOW_SPAN_MM, cubeFitDistance, spanFitDistance, type Arena } from "./bodyCube";
import { createSandbox, type Sandbox } from "./bodySandbox";
import { createOrbitGate, orbitDefault } from "./orbit";
import { createRenderScheduler } from "./renderScheduler";
import type { Theme } from "./theme";

export interface BodyScene {
  /** Ask for a redraw (the scene only draws on demand or while the orbit runs). */
  invalidate(): void;
  dispose(): void;
  /** Joint angles in degrees (right-hand rule about the rig axis through its pivot). Replaces the WHOLE pose: joints
   *  not listed return to rest; unknown ids and non-finite angles are ignored. Redraws. */
  setPose(angles: Partial<Record<JointId, number>>): void;
  /** The joints this body can move (empty for a body without a rig). */
  jointIds(): JointId[];
  /** One frame of a physics clip: for each leg body (e.g. "lf_tibia") the 4x4 column-major matrix (viewer frame, mm)
   *  that carries its mesh from the neutral pose to its pose in that frame. Replaces the WHOLE frame: leg bodies not
   *  listed return to rest; unknown names and malformed matrices are ignored; null = rest. Redraws. */
  setClipFrame(mats: ReadonlyMap<string, ArrayLike<number>> | null): void;
  /** Whole-fly motion in the world (from a body clip's thorax_world): the root transform is applied to every part of the
   *  body, a faint floor grid appears under it (fixed in the world) and the camera follows the thorax smoothly with no
   *  auto-orbit. null = no world: the body at the origin, the grid hidden, the camera glides back. Redraws. */
  setWorld(world: WorldPose | null): void;
  /** The glass-cube sandbox: a transparent cube with a floor grid and the fly inside it. On: the cube is shown, the
   *  loose floor grid is replaced by the cube's own, and the camera closes in to follow the fly (the user can zoom out
   *  to the whole cube). Off: back to the plain body view. Redraws. */
  setSandbox(on: boolean): void;
  /** The objects layer of the sandbox (data only): `[{ id, kind: "sugar" | "coffee", pos_mm: [x, y above the floor, z],
   *  size_mm }]` replaces the layer; invalid entries are dropped and listed in `errors`. Objects show while the sandbox is on. */
  setObjects(objects: unknown): { errors: string[] };
  /** The arena the server enforces (null: the fallback cube): the glass cube is drawn from these bounds so the drawn walls
   *  coincide with the enforced ones. Rebuilds the cube (and re-applies the objects) when it changed. */
  setArena(arena: Arena | null): void;
}

/** The leg bodies a clip can move (6 legs x coxa, trochanterfemur, tibia, tarsus1..5), as named in fly-body.json. */
const LEG_BODY = /^[lr][fmh]_(coxa|trochanterfemur|tibia|tarsus[1-5])$/;

const FOV = 40;
/** Default view: 3/4 front-left (the fly's left is +X, its face +Z), slightly above. */
const VIEW = { azimuth: 35, elevation: 15 };
/** Hidden lines are removed by the depth test, so the remaining (front) lines can be stronger than the 0.6 that
 *  kept the old see-through tangle quiet. 0.9 was chosen by looking at renders of the real body (0.75 read too faint). */
const LINE_OPACITY = 0.9;
/** The fly fills about this share of the pane's limiting dimension at the default view. */
const FILL = 0.7;
/** Floor grid: a cell is this share of the body's size; the grid reaches this many cells each way from the thorax. */
const GRID_CELL = 1 / 5;
const GRID_HALF = 24;
const GRID_OPACITY = 0.22;
/** Camera follow: the share of the remaining distance covered per frame, and the distance (mm) treated as arrived. */
const FOLLOW_ALPHA = 0.18;
const FOLLOW_EPS = 1e-4;

/** The static fly body as light line segments on the page's own background. Throws a clear Error without WebGL. */
const HEAD_CARRIED = new Set<string>(["antenna_l", "antenna_r", "proboscis"]);

export function createBodyScene(
  container: HTMLElement,
  body: ParsedBody,
  theme: Theme,
  opts: { reducedMotion?: boolean } = {},
): BodyScene {
  let renderer: THREE.WebGLRenderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch (e) {
    throw new Error(`the fly body needs WebGL, which is not available: ${(e as Error).message}`);
  }
  // Everything created from here on registers its release in `undo`: when the setup throws half way, the
  // WebGL context, the canvas and the observers are given back instead of leaking until the page closes.
  const undo: Array<() => void> = [() => { renderer.dispose(); renderer.domElement.remove(); }];
  try {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0); // transparent: the CSS vignette shows through
    container.appendChild(renderer.domElement);
    return assembleBodyScene(renderer, undo, container, body, theme, opts);
  } catch (e) {
    for (const release of undo.reverse()) {
      try { release(); } catch { /* keep releasing the rest */ }
    }
    throw e;
  }
}

function assembleBodyScene(
  renderer: THREE.WebGLRenderer,
  undo: Array<() => void>,
  container: HTMLElement,
  body: ParsedBody,
  theme: Theme,
  opts: { reducedMotion?: boolean },
): BodyScene {
  const scene = new THREE.Scene();
  // depthTest on: the occluder mesh below writes depth, so edges on the far side of the body are hidden
  const material = new THREE.LineBasicMaterial({ color: theme.highlightHex, transparent: true, opacity: LINE_OPACITY, depthTest: true, depthWrite: false });
  undo.push(() => material.dispose());
  // Hidden-line occluder: the fly's surface writes depth but NO colour (the canvas stays transparent), and is
  // drawn first. The polygon offset pushes it slightly back so the edges lying on the surface do not z-fight.
  const meshMaterial = body.mesh
    ? new THREE.MeshBasicMaterial({
        colorWrite: false,
        depthWrite: true,
        side: THREE.DoubleSide,
        // Edges lie ON the surface, so they z-fight with the occluder. WebGL polygon offset moves only filled polygons
        // (never lines), so the occluder is pushed back instead. 1/1 lost many edges; 4/4 kept more without letting
        // the far legs show through (checked by eye on renders of the real asset).
        polygonOffset: true,
        polygonOffsetFactor: 4,
        polygonOffsetUnits: 4,
      })
    : null;
  if (meshMaterial) undo.push(() => meshMaterial.dispose());
  const meshPositions = body.mesh ? new THREE.BufferAttribute(body.mesh.positions, 3) : null;

  /** One group holding the lines and the depth-only occluder of one piece of the body. */
  const addGroup = (lines: Float32Array, indices: Uint16Array | Uint32Array | null, name = ""): THREE.Group => {
    const group = new THREE.Group();
    group.name = name;
    if (lines.length > 0) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(lines, 3));
      undo.push(() => g.dispose());
      const obj = new THREE.LineSegments(g, material);
      obj.renderOrder = 0;
      obj.frustumCulled = false; // bounds are those of the rest pose; a joint moves out of them
      group.add(obj);
    }
    if (meshMaterial && meshPositions && indices && indices.length > 0) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", meshPositions);
      g.setIndex(new THREE.BufferAttribute(indices, 1));
      undo.push(() => g.dispose());
      const occluder = new THREE.Mesh(g, meshMaterial);
      occluder.renderOrder = -1;
      occluder.frustumCulled = false;
      group.add(occluder);
    }
    scene.add(group);
    return group;
  };

  // Without a per-part mesh split the pieces cannot move the occluder with their lines: stay one static group.
  const splittable = !body.mesh || (body.meshParts ?? []).length > 0;
  const joints = splittable ? (body.joints ?? []) : [];
  // every leg body a rig joint does not already move gets a group of its own, still at rest (identity) until a clip frame arrives
  const claimed = new Set(joints.flatMap((j) => j.parts));
  const legNames = splittable ? (body.parts ?? []).map((p) => p.name).filter((n) => LEG_BODY.test(n) && !claimed.has(n)) : [];
  const rigged: Array<{ id: JointId; axis: Vec3; pivot: Vec3; group: THREE.Group }> = [];
  const legGroups = new Map<string, THREE.Group>();
  let restGroup: THREE.Group;
  if (joints.length === 0 && legNames.length === 0) {
    restGroup = addGroup(body.positions, body.mesh?.indices ?? null, "rest");
  } else {
    const split = splitByJoint({
      parts: body.parts, meshParts: body.meshParts ?? [],
      joints: [...joints.map((j) => ({ id: j.id as string, parts: j.parts })), ...legNames.map((n) => ({ id: n, parts: [n] }))],
      positions: body.positions, indices: body.mesh?.indices ?? null,
    });
    restGroup = addGroup(split.rest.lines, split.rest.indices, "rest");
    for (const j of joints) {
      const piece = split.joints[j.id];
      const group = addGroup(piece.lines, piece.indices, j.id);
      group.matrixAutoUpdate = false; // the matrix below is the whole transform: vertices stay at their asset positions
      rigged.push({ id: j.id, axis: j.axis, pivot: j.pivot, group });
    }
    for (const n of legNames) {
      const piece = split.joints[n];
      const group = addGroup(piece.lines, piece.indices, n);
      group.matrixAutoUpdate = false;
      legGroups.set(n, group);
    }
  }

  restGroup.matrixAutoUpdate = false; // like every group: its matrix is the world transform alone

  const size = body.radius * 2; // farthest vertex from the centroid, doubled
  const camera = new THREE.PerspectiveCamera(FOV, 1, size / 200, size * 20);
  const target = new THREE.Vector3(...body.centroid);
  let invalidate = () => {};

  // Floor: a faint grid at the body's lowest point, in the page's line colour, fixed in the world (it is re-centred in
  // whole cells under the thorax, so it reads as still while the fly moves over it). Hidden until a clip has a world.
  const cell = size * GRID_CELL;
  const floorGeometry = new THREE.BufferGeometry();
  floorGeometry.setAttribute("position", new THREE.BufferAttribute(gridSegments(GRID_HALF, cell), 3));
  const floorMaterial = new THREE.LineBasicMaterial({ color: theme.highlightHex, transparent: true, opacity: GRID_OPACITY, depthWrite: false });
  undo.push(() => { floorGeometry.dispose(); floorMaterial.dispose(); });
  const floor = new THREE.LineSegments(floorGeometry, floorMaterial);
  floor.name = "floor";
  floor.renderOrder = -2;
  floor.frustumCulled = false;
  floor.visible = false;
  floor.position.y = body.bounds?.min?.[1] ?? 0;
  scene.add(floor);

  // built on first use: a viewer that never opens the sandbox pays nothing for it
  let sandboxMade: Sandbox | null = null;
  let arenaNow: Arena | null = null;
  let objectsRaw: unknown = [];
  const sandbox = (): Sandbox => {
    if (!sandboxMade) {
      const made = createSandbox(theme, body.bounds?.min?.[1] ?? 0, undefined, arenaNow);
      undo.push(() => made.dispose());
      scene.add(made.group);
      sandboxMade = made;
    }
    return sandboxMade;
  };
  let sandboxOn = false;

  const controls = new OrbitControls(camera, renderer.domElement);
  undo.push(() => controls.dispose());
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.target.copy(target);
  controls.minDistance = size * 0.25;
  controls.maxDistance = size * 4;
  controls.autoRotateSpeed = theme.orbitSpeed;
  const gate = createOrbitGate(orbitDefault(opts.reducedMotion === true), theme.orbitResumeMs);
  controls.addEventListener("start", () => { gate.press(); invalidate(); });
  controls.addEventListener("end", () => { gate.release(performance.now()); invalidate(); });
  controls.addEventListener("change", () => invalidate());

  const place = (aspect: number) => {
    const d = viewFitDistance(body.positions, [target.x, target.y, target.z], VIEW.azimuth, VIEW.elevation, FOV, aspect, FILL);
    const az = (VIEW.azimuth * Math.PI) / 180;
    const el = (VIEW.elevation * Math.PI) / 180;
    camera.position.set(
      target.x + d * Math.sin(az) * Math.cos(el),
      target.y + d * Math.sin(el),
      target.z + d * Math.cos(az) * Math.cos(el),
    );
    controls.update();
  };

  let placed = false;
  const resize = () => {
    const w = container.clientWidth || window.innerWidth;
    const h = container.clientHeight || window.innerHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    if (!placed) { place(camera.aspect); placed = true; } // frame once; later resizes keep the user's view
    invalidate();
  };
  const ro = new ResizeObserver(resize);
  undo.push(() => ro.disconnect());
  ro.observe(container);
  resize();

  // Camera follow: the focus (and the camera with it, so the view keeps its angle and distance) glides toward the thorax.
  let world: WorldPose | null = null;
  const defaultFar = camera.far;
  /** Move the camera along its line of sight to `distance` from the focus. */
  const setDistance = (distance: number) => {
    const dir = camera.position.clone().sub(controls.target);
    if (dir.lengthSq() === 0) dir.set(0, 0, 1);
    camera.position.copy(controls.target).addScaledVector(dir.normalize(), distance);
    controls.update();
  };
  const frameCube = (box: Sandbox) => {
    const whole = cubeFitDistance(box.bounds, FOV, camera.aspect);
    camera.far = Math.max(defaultFar, whole * 3);
    controls.maxDistance = Math.max(size * 4, whole * 1.05);
    setDistance(spanFitDistance(FOLLOW_SPAN_MM, FOV, camera.aspect));
  };
  const sameArena = (a: Arena | null, b: Arena | null) => a === b || (a !== null && b !== null && a.interiorMm === b.interiorMm && a.boundsMm.every((v, i) => v === b.boundsMm[i]));
  const setArena = (a: Arena | null) => {
    if (sameArena(a, arenaNow)) return;
    arenaNow = a;
    const old = sandboxMade;
    if (!old) return; // built lazily with the arena
    scene.remove(old.group);
    old.dispose();
    sandboxMade = null;
    const box = sandbox();
    box.setVisible(sandboxOn);
    box.setObjects(objectsRaw);
    if (sandboxOn) { frameCube(box); camera.updateProjectionMatrix(); }
    invalidate();
  };
  const setSandbox = (on: boolean) => {
    if (on === sandboxOn) return;
    sandboxOn = on;
    const box = sandbox();
    box.setVisible(on);
    floor.visible = world !== null && !on;
    if (on) {
      frameCube(box);
    } else {
      camera.far = defaultFar;
      controls.maxDistance = size * 4;
      setDistance(viewFitDistance(body.positions, [target.x, target.y, target.z], VIEW.azimuth, VIEW.elevation, FOV, camera.aspect, FILL));
    }
    camera.updateProjectionMatrix();
    invalidate();
  };
  const focus: [number, number, number] = [0, 0, 0];
  const wanted: [number, number, number] = [0, 0, 0];
  const follow = (): boolean => {
    const r = followStep(focus, wanted, opts.reducedMotion ? 1 : FOLLOW_ALPHA, FOLLOW_EPS);
    const d = new THREE.Vector3(r.pos[0] - focus[0], r.pos[1] - focus[1], r.pos[2] - focus[2]);
    focus[0] = r.pos[0]; focus[1] = r.pos[1]; focus[2] = r.pos[2];
    if (d.lengthSq() > 0) { camera.position.add(d); controls.target.add(d); }
    return !r.settled;
  };

  const scheduler = createRenderScheduler({
    schedule: (cb) => requestAnimationFrame(cb),
    cancel: (id) => cancelAnimationFrame(id),
    render: () => {
      const orbiting = gate.active(performance.now()) && world === null; // no auto-orbit while following the fly
      controls.autoRotate = orbiting;
      const gliding = follow();
      const moved = controls.update();
      renderer.render(scene, camera);
      // gate.enabled keeps the loop alive through the pause after a drag so the orbit can resume on its own
      return moved || orbiting || gate.enabled || gliding;
    },
  });
  undo.push(() => scheduler.dispose());
  invalidate = () => scheduler.invalidate();
  invalidate();

  // Every group's matrix = world * its own motion (a rig joint's rotation, a leg body's clip matrix). With no world the
  // own motion is used as it is.
  const worldMatrix = new THREE.Matrix4();
  const own = new Map<THREE.Group, THREE.Matrix4>();
  const placeGroup = (group: THREE.Group) => {
    const m = own.get(group);
    if (world === null) group.matrix.copy(m ?? new THREE.Matrix4());
    else group.matrix.copy(worldMatrix).multiply(m ?? new THREE.Matrix4());
    group.matrixWorldNeedsUpdate = true;
  };

  const setPose = (angles: Partial<Record<JointId, number>>) => {
    // the head yaws about its neck pivot and carries the antennae and the proboscis with it: their own rotation is applied first, then the head's
    const headJ = rigged.find((r) => r.id === "head");
    const headDeg = angles.head;
    const headM = headJ && typeof headDeg === "number" && Number.isFinite(headDeg)
      ? new THREE.Matrix4().fromArray(rotationMatrix(headJ.axis, (headDeg * Math.PI) / 180, headJ.pivot)) : null;
    for (const j of rigged) {
      if (j.id === "head") {
        own.set(j.group, headM ?? new THREE.Matrix4());
        placeGroup(j.group);
        continue;
      }
      const deg = angles[j.id];
      const rad = typeof deg === "number" && Number.isFinite(deg) ? (deg * Math.PI) / 180 : 0;
      const fl = (k: string) => { const v = (angles as Record<string, number | undefined>)[`${j.id}_${k}`]; return typeof v === "number" && Number.isFinite(v) ? (v * Math.PI) / 180 : null; };
      const [st, dv, ro] = j.id === "wing_l" || j.id === "wing_r" ? [fl("stroke"), fl("dev"), fl("rot")] : [null, null, null];
      if (st !== null || dv !== null || ro !== null) { // flight wing: R_up(stroke) R_fore(deviation) R_span(rotation) about the hinge (viewer axes)
        const r = (axis: Vec3, a: number | null) => new THREE.Matrix4().fromArray(rotationMatrix(axis, a ?? 0, j.pivot));
        own.set(j.group, r([0, 1, 0], st).multiply(r([0, 0, 1], dv)).multiply(r([1, 0, 0], ro)));
        placeGroup(j.group);
        continue;
      }
      const jm = new THREE.Matrix4().fromArray(rotationMatrix(j.axis, rad, j.pivot));
      own.set(j.group, headM && HEAD_CARRIED.has(j.id) ? headM.clone().multiply(jm) : jm);
      placeGroup(j.group);
    }
    invalidate();
  };

  const setClipFrame = (mats: ReadonlyMap<string, ArrayLike<number>> | null) => {
    for (const [name, group] of legGroups) {
      const m = mats?.get(name);
      if (m && m.length === 16 && Array.from(m).every(Number.isFinite)) own.set(group, new THREE.Matrix4().fromArray(Array.from(m)));
      else own.delete(group);
      placeGroup(group);
    }
    invalidate();
  };

  const setWorld = (w: WorldPose | null) => {
    const ok = w !== null && w.matrix.length === 16 && w.matrix.every(Number.isFinite) && w.offset.every(Number.isFinite);
    world = ok ? w : null;
    if (world) worldMatrix.fromArray(world.matrix);
    wanted[0] = world?.offset[0] ?? 0; wanted[1] = world?.offset[1] ?? 0; wanted[2] = world?.offset[2] ?? 0;
    for (const g of [restGroup, ...rigged.map((j) => j.group), ...legGroups.values()]) placeGroup(g);
    floor.visible = world !== null && !sandboxOn;
    if (world) {
      const [gx, gz] = gridSnap(wanted, cell);
      floor.position.x = gx;
      floor.position.z = gz;
    }
    invalidate();
  };

  return {
    invalidate: () => invalidate(),
    setPose,
    setClipFrame,
    setWorld,
    setSandbox,
    setObjects: (objects) => { objectsRaw = objects; const r = sandbox().setObjects(objects); invalidate(); return r; },
    setArena,
    jointIds: () => rigged.map((j) => j.id),
    dispose() {
      // the undo list holds every resource in creation order: scheduler, observer, controls, geometries, materials, renderer
      for (const release of undo.splice(0).reverse()) {
        try { release(); } catch { /* keep releasing the rest */ }
      }
    },
  };
}
