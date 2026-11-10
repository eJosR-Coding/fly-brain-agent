import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { Brain } from "./brainData";
import { writeSelectionIndex } from "./highlight";
import type { Theme } from "./theme";

export interface BrainScene {
  /** Light up these row indices on the highlight layer (replaces any previous one); null or [] clears. */
  setHighlight(indices: ArrayLike<number> | null): void;
  dispose(): void;
}

/** Round sprite for gl_PointCoord: a crisp disc, or a disc with a soft glow falloff for highlights. */
function dotTexture(glow: boolean): THREE.CanvasTexture {
  const size = 64;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d")!;
  const r = size / 2;
  const grad = g.createRadialGradient(r, r, 0, r, r, r);
  if (glow) {
    grad.addColorStop(0, "rgba(255,255,255,1)");
    grad.addColorStop(0.45, "rgba(255,255,255,0.9)");
    grad.addColorStop(1, "rgba(255,255,255,0)");
  } else {
    grad.addColorStop(0, "rgba(255,255,255,1)");
    grad.addColorStop(0.8, "rgba(255,255,255,1)");
    grad.addColorStop(1, "rgba(255,255,255,0)");
  }
  g.fillStyle = grad;
  g.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(c);
}

export function createScene(container: HTMLElement, brain: Brain, theme: Theme): BrainScene {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0); // transparent: the vignette behind the canvas comes from CSS
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const maxDim = Math.max(...brain.size);
  scene.fog = new THREE.Fog(theme.fogColor, maxDim * theme.fogNear, maxDim * theme.fogFar);

  const count = brain.meta.count;
  const baseSize = maxDim / theme.pointSizeDivisor;
  const positions = new THREE.BufferAttribute(brain.positions, 3);

  const baseGeometry = new THREE.BufferGeometry();
  baseGeometry.setAttribute("position", positions);
  const baseMaterial = new THREE.PointsMaterial({
    color: theme.baseHex,
    map: dotTexture(false),
    size: baseSize,
    sizeAttenuation: true,
    transparent: true,
    opacity: theme.pointOpacity,
    depthWrite: false,
  });
  const basePoints = new THREE.Points(baseGeometry, baseMaterial);

  // Highlight layer: an indexed Points sharing the position buffer, bigger and additive, no fog.
  const indexBuf = new Uint32Array(count);
  const indexAttr = new THREE.BufferAttribute(indexBuf, 1);
  indexAttr.setUsage(THREE.DynamicDrawUsage);
  const hlGeometry = new THREE.BufferGeometry();
  hlGeometry.setAttribute("position", positions);
  hlGeometry.setIndex(indexAttr);
  hlGeometry.setDrawRange(0, 0);
  const hlMaterial = new THREE.PointsMaterial({
    color: theme.highlightHex,
    map: dotTexture(true),
    size: baseSize * theme.highlightSizeScale,
    sizeAttenuation: true,
    transparent: true,
    opacity: theme.highlightOpacity,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    fog: false,
  });
  const hlPoints = new THREE.Points(hlGeometry, hlMaterial);

  // center the cloud on its bounding box; the wrapper flips y (FlyWire y grows downward)
  const flip = new THREE.Group();
  flip.scale.y = -1;
  for (const pts of [basePoints, hlPoints]) {
    pts.frustumCulled = false;
    pts.position.set(-brain.center[0], -brain.center[1], -brain.center[2]);
    flip.add(pts);
  }
  hlPoints.renderOrder = 1; // after the base layer
  scene.add(flip);

  const camera = new THREE.PerspectiveCamera(45, 1, maxDim / 100, maxDim * 20);
  camera.position.set(0, 0, maxDim * 1.4);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0);

  const resize = () => {
    const w = container.clientWidth || window.innerWidth;
    const h = container.clientHeight || window.innerHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();

  let raf = 0;
  const loop = () => {
    raf = requestAnimationFrame(loop);
    controls.update();
    renderer.render(scene, camera);
  };
  loop();

  return {
    setHighlight(indices) {
      const n = indices ? writeSelectionIndex(indexBuf, indices) : 0;
      hlGeometry.setDrawRange(0, n);
      indexAttr.needsUpdate = true;
    },
    dispose() {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      baseGeometry.dispose();
      hlGeometry.dispose();
      baseMaterial.dispose();
      hlMaterial.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
