import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { Brain } from "./brainData";
import { writeSelectionIndex } from "./highlight";
import type { Theme } from "./theme";
import { homeCamera, labelSpecs, placeLabels, screenBounds, type LabelPlacement, type Landmarks } from "./anatomy";
import { buildRegionColors, regionPalette, hexToRgb, type Regions } from "./regions";
import { createOrbitGate, orbitDefault } from "./orbit";
import { ntPalette, ntWeights, type Transmitters } from "./transmitters";

export interface BrainScene {
  /** Light up these row indices on the highlight layer (replaces any previous one); null or [] clears. */
  setHighlight(indices: ArrayLike<number> | null): void;
  /** Color the base points by region tone or by neurotransmitter tier. Ignored when that layer is unavailable. */
  setColorMode(mode: "regions" | "transmitters"): void;
  /** True when the transmitter layer loaded. */
  readonly hasTransmitters: boolean;
  /** Turn the slow auto-orbit on or off (it also pauses while the user interacts). */
  setOrbit(on: boolean): void;
  readonly orbitOn: boolean;
  /** Move the camera to a preset: "home" is a gentle 3/4 view, "front" looks straight at the face. */
  setView(view: "home" | "front"): void;
  dispose(): void;
}

/** Optional context: region codes tint the points, landmarks drive labels. */
export interface Anatomy {
  regions?: Regions;
  nt?: Transmitters;
  landmarks?: Landmarks;
}

/** Camera distance as a multiple of the brain's largest dimension: leaves room for labels on both sides. */
const FIT = 1.55;
const HOME_VIEW = { azimuth: 16, elevation: 10 };

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

export function createScene(container: HTMLElement, brain: Brain, theme: Theme, anatomy: Anatomy = {}): BrainScene {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0); // transparent: the vignette behind the canvas comes from CSS
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const maxDim = Math.max(...brain.size);
  // radius of the sphere around the cloud: the depth cue spans the camera distance +- this
  const radius = Math.hypot(...brain.size) / 2;

  const count = brain.meta.count;
  const baseSize = maxDim / theme.pointSizeDivisor;
  const positions = new THREE.BufferAttribute(brain.positions, 3);

  // Per-point color and size multiplier live in dynamic attributes so the color mode can swap them.
  const flat = new Float32Array(count * 3);
  const [fr, fg, fb] = hexToRgb(theme.baseHex);
  for (let i = 0; i < count; i++) flat.set([fr, fg, fb], i * 3);
  const regionColors = anatomy.regions
    ? buildRegionColors(anatomy.regions.codes, regionPalette(anatomy.regions.table, theme.tones, theme.baseHex))
    : flat;
  let ntColors: Float32Array | null = null;
  let ntSizes: Float32Array | null = null;
  if (anatomy.nt) {
    const { palette, sizes } = ntPalette(anatomy.nt.table);
    ntColors = buildRegionColors(anatomy.nt.codes, palette);
    ntSizes = ntWeights(anatomy.nt.codes, sizes);
  }
  const ones = new Float32Array(count).fill(1);
  const colorAttr = new THREE.BufferAttribute(new Float32Array(regionColors), 3);
  const weightAttr = new THREE.BufferAttribute(new Float32Array(ones), 1);
  colorAttr.setUsage(THREE.DynamicDrawUsage);
  weightAttr.setUsage(THREE.DynamicDrawUsage);

  const baseGeometry = new THREE.BufferGeometry();
  baseGeometry.setAttribute("position", positions);
  baseGeometry.setAttribute("aColor", colorAttr);
  baseGeometry.setAttribute("aWeight", weightAttr);
  const dot = dotTexture(false);
  // Depth cue in the vertex shader: nearer points are brighter and larger, far ones dimmer and smaller.
  const baseMaterial = new THREE.ShaderMaterial({
    uniforms: {
      uMap: { value: dot },
      uSize: { value: baseSize },
      uScale: { value: 1 },
      uNear: { value: 0 },
      uFar: { value: 1 },
      uOpacity: { value: theme.pointOpacity },
      uGain: { value: new THREE.Vector2(theme.depthGainFar, theme.depthGainNear) },
      uGrow: { value: new THREE.Vector2(theme.depthSizeFar, theme.depthSizeNear) },
    },
    vertexShader: `
      attribute vec3 aColor;
      attribute float aWeight;
      uniform float uSize, uScale, uNear, uFar;
      uniform vec2 uGain, uGrow;
      varying vec3 vColor;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        float depth = max(-mv.z, 1e-4);
        float t = clamp((uFar - depth) / max(uFar - uNear, 1e-4), 0.0, 1.0); // 1 = nearest
        vColor = aColor * mix(uGain.x, uGain.y, t);
        gl_PointSize = max(uSize * uScale * mix(uGrow.x, uGrow.y, t) * aWeight / depth, 1.0);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      uniform sampler2D uMap;
      uniform float uOpacity;
      varying vec3 vColor;
      void main() {
        float a = texture2D(uMap, gl_PointCoord).a * uOpacity;
        gl_FragColor = vec4(vColor, a);
        #include <colorspace_fragment>
      }`,
    transparent: true,
    depthWrite: false,
  });
  const basePoints = new THREE.Points(baseGeometry, baseMaterial);

  // Highlight layer: an indexed Points sharing the position buffer, bigger and additive, drawn on top of the base layer.
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

  // Anatomy context: labels are HTML projected each frame from landmark centroids.
  const world = new THREE.Group();
  world.position.set(-brain.center[0], -brain.center[1], -brain.center[2]);
  flip.add(world);

  const labelsEl = document.createElement("div");
  labelsEl.id = "labels";
  container.appendChild(labelsEl);
  type LabelState = { el: HTMLElement; anchor: THREE.Vector3; kind: "side" | "top"; side: LabelPlacement["side"] | "" };
  const labels: LabelState[] = (anatomy.landmarks ? labelSpecs(anatomy.landmarks) : []).map((spec) => {
    const el = document.createElement("div");
    el.className = "lm";
    el.append(document.createElement("i"), Object.assign(document.createElement("span"), { textContent: spec.text }));
    labelsEl.appendChild(el);
    return { el, anchor: new THREE.Vector3(...spec.position), kind: spec.kind, side: "" };
  });
  // A sparse sample of the cloud, projected each frame, gives the silhouette the labels must stay outside of.
  const SAMPLE = 4000;
  const stride = Math.max(1, Math.floor(count / SAMPLE));
  const sample: number[] = [];
  for (let i = 0; i < count; i += stride) sample.push(i);
  const tmp = new THREE.Vector3(); // reused every frame: no per-frame allocation
  const mid = new THREE.Vector3();
  const xy = new Float32Array(sample.length * 2);
  const LABEL = { margin: 26, spacing: 17 };
  const placeAnatomyLabels = (w: number, h: number) => {
    if (!labels.length) return;
    for (let k = 0; k < sample.length; k++) {
      const i = sample[k] * 3;
      tmp.set(brain.positions[i], brain.positions[i + 1], brain.positions[i + 2]).applyMatrix4(world.matrixWorld).project(camera);
      xy[k * 2] = ((tmp.x + 1) / 2) * w;
      xy[k * 2 + 1] = ((1 - tmp.y) / 2) * h;
    }
    const box = screenBounds(xy);
    if (!box) return;
    mid.set(brain.center[0], brain.center[1], brain.center[2]).applyMatrix4(world.matrixWorld).project(camera);
    const centerX = ((mid.x + 1) / 2) * w;
    const visible: { l: LabelState; ax: number; ay: number }[] = [];
    for (const l of labels) {
      tmp.copy(l.anchor).applyMatrix4(world.matrixWorld).project(camera);
      const behind = tmp.z > 1;
      l.el.classList.toggle("hidden", behind);
      if (!behind) visible.push({ l, ax: ((tmp.x + 1) / 2) * w, ay: ((1 - tmp.y) / 2) * h });
    }
    const placed = placeLabels(visible.map((v) => ({ ax: v.ax, ay: v.ay, kind: v.l.kind })), box, centerX, { ...LABEL, width: w, height: h });
    visible.forEach((v, i) => {
      const p = placed[i];
      if (v.l.side !== p.side) { v.l.side = p.side; v.l.el.className = `lm ${p.side}`; }
      const st = v.l.el.style;
      st.setProperty("--dx", `${p.dx}px`);
      st.setProperty("--dy", `${p.dy}px`);
      st.setProperty("--len", `${p.length}px`);
      st.setProperty("--ang", `${p.angle}deg`);
      st.transform = `translate(${v.ax}px, ${v.ay}px)`;
    });
  };

  const camera = new THREE.PerspectiveCamera(45, 1, maxDim / 100, maxDim * 20);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.autoRotateSpeed = theme.orbitSpeed;
  const reduced = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;
  const gate = createOrbitGate(orbitDefault(reduced), theme.orbitResumeMs);
  controls.addEventListener("start", () => gate.press());
  controls.addEventListener("end", () => gate.release(performance.now()));
  const setView = (view: "home" | "front") => {
    const [x, y, z] = view === "front" ? homeCamera(maxDim * FIT, 0, 0) : homeCamera(maxDim * FIT, HOME_VIEW.azimuth, HOME_VIEW.elevation);
    camera.position.set(x, y, z);
    controls.target.set(0, 0, 0);
    gate.touch(performance.now());
    controls.update();
  };
  setView("home");

  const resize = () => {
    const w = container.clientWidth || window.innerWidth;
    const h = container.clientHeight || window.innerHeight;
    renderer.setSize(w, h);
    baseMaterial.uniforms.uScale.value = renderer.getDrawingBufferSize(new THREE.Vector2()).y / 2;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();

  let raf = 0;
  const loop = () => {
    raf = requestAnimationFrame(loop);
    controls.autoRotate = gate.active(performance.now());
    controls.update();
    const dist = camera.position.distanceTo(controls.target);
    baseMaterial.uniforms.uNear.value = dist - radius;
    baseMaterial.uniforms.uFar.value = dist + radius;
    renderer.render(scene, camera);
    placeAnatomyLabels(container.clientWidth, container.clientHeight);
  };
  loop();

  return {
    setHighlight(indices) {
      const n = indices ? writeSelectionIndex(indexBuf, indices) : 0;
      hlGeometry.setDrawRange(0, n);
      indexAttr.needsUpdate = true;
    },
    setColorMode(mode) {
      const useNt = mode === "transmitters" && ntColors !== null && ntSizes !== null;
      colorAttr.array.set(useNt ? ntColors! : regionColors);
      weightAttr.array.set(useNt ? ntSizes! : ones);
      colorAttr.needsUpdate = true;
      weightAttr.needsUpdate = true;
    },
    hasTransmitters: ntColors !== null,
    setOrbit(on) {
      gate.setEnabled(on);
    },
    get orbitOn() {
      return gate.enabled;
    },
    setView,
    dispose() {
      cancelAnimationFrame(raf);
      labelsEl.remove();
      ro.disconnect();
      controls.dispose();
      baseGeometry.dispose();
      hlGeometry.dispose();
      baseMaterial.dispose();
      dot.dispose();
      hlMaterial.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
