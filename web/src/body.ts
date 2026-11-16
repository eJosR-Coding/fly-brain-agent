import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { viewFitDistance, type ParsedBody, type Vec3 } from "./bodyData";
import type { JointId } from "./bodyPose";
import { rotationMatrix, splitByJoint } from "./bodyRig";
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
}

const FOV = 40;
/** Default view: 3/4 front-left (the fly's left is +X, its face +Z), slightly above. */
const VIEW = { azimuth: 35, elevation: 15 };
/** Hidden lines are removed by the depth test, so the remaining (front) lines can be stronger than the 0.6 that
 *  kept the old see-through tangle quiet. 0.9 was chosen by looking at renders of the real body (0.75 read too faint). */
const LINE_OPACITY = 0.9;
/** The fly fills about this share of the pane's limiting dimension at the default view. */
const FILL = 0.7;

/** The static fly body as light line segments on the page's own background. Throws a clear Error without WebGL. */
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
  const addGroup = (lines: Float32Array, indices: Uint16Array | Uint32Array | null): THREE.Group => {
    const group = new THREE.Group();
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

  // Without a per-part mesh split the joints cannot move the occluder with their lines: stay one static group.
  const joints = (body.joints ?? []).filter(() => !body.mesh || (body.meshParts ?? []).length > 0);
  const rigged: Array<{ id: JointId; axis: Vec3; pivot: Vec3; group: THREE.Group }> = [];
  if (joints.length === 0) {
    addGroup(body.positions, body.mesh?.indices ?? null);
  } else {
    const split = splitByJoint({
      parts: body.parts, meshParts: body.meshParts ?? [], joints, positions: body.positions, indices: body.mesh?.indices ?? null,
    });
    addGroup(split.rest.lines, split.rest.indices);
    for (const j of joints) {
      const piece = split.joints[j.id];
      const group = addGroup(piece.lines, piece.indices);
      group.matrixAutoUpdate = false; // the matrix below is the whole transform: vertices stay at their asset positions
      rigged.push({ id: j.id, axis: j.axis, pivot: j.pivot, group });
    }
  }

  const size = body.radius * 2; // farthest vertex from the centroid, doubled
  const camera = new THREE.PerspectiveCamera(FOV, 1, size / 200, size * 20);
  const target = new THREE.Vector3(...body.centroid);
  let invalidate = () => {};

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

  const scheduler = createRenderScheduler({
    schedule: (cb) => requestAnimationFrame(cb),
    cancel: (id) => cancelAnimationFrame(id),
    render: () => {
      const orbiting = gate.active(performance.now());
      controls.autoRotate = orbiting;
      const moved = controls.update();
      renderer.render(scene, camera);
      // gate.enabled keeps the loop alive through the pause after a drag so the orbit can resume on its own
      return moved || orbiting || gate.enabled;
    },
  });
  undo.push(() => scheduler.dispose());
  invalidate = () => scheduler.invalidate();
  invalidate();

  const setPose = (angles: Partial<Record<JointId, number>>) => {
    for (const j of rigged) {
      const deg = angles[j.id];
      const rad = typeof deg === "number" && Number.isFinite(deg) ? (deg * Math.PI) / 180 : 0;
      j.group.matrix.fromArray(rotationMatrix(j.axis, rad, j.pivot));
      j.group.matrixWorldNeedsUpdate = true;
    }
    invalidate();
  };

  return {
    invalidate: () => invalidate(),
    setPose,
    jointIds: () => rigged.map((j) => j.id),
    dispose() {
      // the undo list holds every resource in creation order: scheduler, observer, controls, geometries, materials, renderer
      for (const release of undo.splice(0).reverse()) {
        try { release(); } catch { /* keep releasing the rest */ }
      }
    },
  };
}
