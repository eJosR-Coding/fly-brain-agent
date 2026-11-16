import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { viewFitDistance, type ParsedBody } from "./bodyData";
import { createOrbitGate, orbitDefault } from "./orbit";
import { createRenderScheduler } from "./renderScheduler";
import type { Theme } from "./theme";

export interface BodyScene {
  /** Ask for a redraw (the scene only draws on demand or while the orbit runs). */
  invalidate(): void;
  dispose(): void;
}

const FOV = 40;
/** Default view: 3/4 front-left (the fly's left is +X, its face +Z), slightly above. */
const VIEW = { azimuth: 35, elevation: 15 };
const LINE_OPACITY = 0.6;
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
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0); // transparent: the CSS vignette shows through
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(body.positions, 3));
  const material = new THREE.LineBasicMaterial({ color: theme.highlightHex, transparent: true, opacity: LINE_OPACITY, depthWrite: false });
  scene.add(new THREE.LineSegments(geometry, material));

  const size = body.radius * 2; // farthest vertex from the centroid, doubled
  const camera = new THREE.PerspectiveCamera(FOV, 1, size / 200, size * 20);
  const target = new THREE.Vector3(...body.centroid);
  let invalidate = () => {};

  const controls = new OrbitControls(camera, renderer.domElement);
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
  invalidate = () => scheduler.invalidate();
  invalidate();

  return {
    invalidate: () => invalidate(),
    dispose() {
      scheduler.dispose();
      ro.disconnect();
      controls.dispose();
      geometry.dispose();
      material.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
