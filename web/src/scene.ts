import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { Brain } from "./brainData";
import { baseColors } from "./colors";
import type { Theme } from "./theme";

export interface BrainScene {
  colors: Float32Array;
  base: Float32Array;
  /** call after mutating `colors` so the GPU buffer refreshes */
  refreshColors(): void;
  dispose(): void;
}

export function createScene(container: HTMLElement, brain: Brain, theme: Theme): BrainScene {
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(new THREE.Color(theme.background), 1);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const maxDim = Math.max(...brain.size);

  const base = baseColors(brain.meta.count, theme.baseRgb);
  const colors = base.slice();
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(brain.positions, 3));
  const colorAttr = new THREE.BufferAttribute(colors, 3);
  colorAttr.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute("color", colorAttr);

  const material = new THREE.PointsMaterial({
    vertexColors: true,
    size: maxDim / theme.pointSizeDivisor,
    sizeAttenuation: true,
    transparent: true,
    opacity: theme.pointOpacity,
    depthWrite: false,
  });
  const points = new THREE.Points(geometry, material);
  points.frustumCulled = false;
  // center the cloud on its bounding box; the wrapper flips y (FlyWire y grows downward)
  points.position.set(-brain.center[0], -brain.center[1], -brain.center[2]);
  const flip = new THREE.Group();
  flip.scale.y = -1;
  flip.add(points);
  scene.add(flip);

  const camera = new THREE.PerspectiveCamera(45, 1, maxDim / 100, maxDim * 20);
  camera.position.set(0, 0, maxDim * 1.2);
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
    colors,
    base,
    refreshColors() {
      colorAttr.needsUpdate = true;
    },
    dispose() {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      geometry.dispose();
      material.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
