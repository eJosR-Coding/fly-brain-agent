/** The cables of a find_path result: glowing segments, bright nodes, hop labels and a traveling light. */
import * as THREE from "three";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { buildHops, pointAlong, routeVertices, type PathPayload } from "./path";
import { dotTexture } from "./textures";

export interface PathLayer {
  /** Draw this route (replacing any previous one); null removes it. */
  set(path: PathPayload | null): void;
  /** Per frame, after the scene rendered: move the traveling light and the hop labels. */
  update(now: number, camera: THREE.Camera, width: number, height: number): void;
  setResolution(width: number, height: number): void;
  dispose(): void;
}

export interface PathLayerOptions {
  /** group (already centered like the cloud) the cables are added to */
  parent: THREE.Object3D;
  /** raw neuron positions (xyz, atlas row order) */
  positions: ArrayLike<number>;
  /** container for the HTML hop labels */
  labels: HTMLElement;
  /** no traveling light, everything static */
  reducedMotion: boolean;
}

/** Seconds the light takes from source to target, and how many lights ride the route at once. */
const FLOW_SECONDS = 3.2;
const LIGHTS = 3;
const GLOW_WIDTH = 4; // the faint underlay is this many times wider than the core cable

export function createPathLayer({ parent, positions, labels, reducedMotion }: PathLayerOptions): PathLayer {
  const group = new THREE.Group();
  group.renderOrder = 4;
  parent.add(group);
  const glow = dotTexture(true);
  const materials: LineMaterial[] = [];
  const geometries: THREE.BufferGeometry[] = [];
  const nodeMaterials: THREE.PointsMaterial[] = [];
  let resolution = new THREE.Vector2(1, 1);
  let route: THREE.Vector3[] = [];
  let lights: THREE.Points | null = null;
  let hopLabels: { el: HTMLElement; mid: THREE.Vector3 }[] = [];
  let routeRaw: [number, number, number][] = [];
  const tmp = new THREE.Vector3();

  const clear = () => {
    group.clear();
    for (const m of materials) m.dispose();
    for (const g of geometries) g.dispose();
    for (const m of nodeMaterials) m.dispose();
    materials.length = geometries.length = nodeMaterials.length = 0;
    for (const l of hopLabels) l.el.remove();
    hopLabels = [];
    lights = null;
    route = [];
    routeRaw = [];
  };

  const cable = (from: number[], to: number[], width: number, opacity: number) => {
    const geo = new LineSegmentsGeometry();
    geo.setPositions([...from, ...to]);
    const mat = new LineMaterial({
      color: 0xffffff, linewidth: width, transparent: true, opacity, blending: THREE.AdditiveBlending,
      depthTest: false, depthWrite: false,
      side: THREE.DoubleSide, // the stage is mirrored in y (negative scale): front faces would be culled
    });
    mat.resolution.copy(resolution);
    const line = new LineSegments2(geo, mat);
    line.frustumCulled = false;
    line.renderOrder = 4;
    group.add(line);
    geometries.push(geo);
    materials.push(mat);
  };

  const dots = (pts: [number, number, number][], sizes: number[], opacity: number) => {
    // one PointsMaterial has one size: draw each size class as its own Points
    const made: THREE.Points[] = [];
    const unique = [...new Set(sizes)];
    for (const px of unique) {
      const idx = sizes.flatMap((s, i) => (s === px ? [i] : []));
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(idx.flatMap((i) => pts[i])), 3));
      const mat = new THREE.PointsMaterial({
        map: glow, size: px, sizeAttenuation: false, transparent: true, opacity, blending: THREE.AdditiveBlending,
        depthTest: false, depthWrite: false, fog: false,
      });
      const p = new THREE.Points(g, mat);
      p.frustumCulled = false;
      p.renderOrder = 5;
      group.add(p);
      geometries.push(g);
      nodeMaterials.push(mat);
      made.push(p);
    }
    return made;
  };

  return {
    set(path) {
      clear();
      if (!path) return;
      const hops = buildHops(path, positions);
      for (const h of hops) {
        cable(h.from, h.to, h.width * GLOW_WIDTH, 0.22); // soft halo
        cable(h.from, h.to, h.width, 0.9); // the core
      }
      routeRaw = routeVertices(path, positions);
      route = routeRaw.map((v) => new THREE.Vector3(...v));
      const last = routeRaw.length - 1;
      // nodes: source and target stand out; the neurons in between are smaller
      dots(routeRaw, routeRaw.map((_, i) => (i === 0 || i === last ? 18 : 12)), 1);
      hopLabels = hops.map((h, i) => {
        const el = document.createElement("div");
        el.className = i % 2 ? "path-label below" : "path-label"; // alternate sides so neighbours do not collide
        el.textContent = h.label;
        labels.appendChild(el);
        return { el, mid: new THREE.Vector3(...h.mid) };
      });
      if (!reducedMotion) {
        const [pts] = dots(new Array(LIGHTS).fill(routeRaw[0]), new Array(LIGHTS).fill(26), 1);
        lights = pts;
      }
    },
    update(now, camera, width, height) {
      if (!route.length) return;
      if (lights) {
        const attr = lights.geometry.getAttribute("position") as THREE.BufferAttribute;
        for (let k = 0; k < attr.count; k++) {
          const t = (((now / 1000) / FLOW_SECONDS + k / LIGHTS) % 1 + 1) % 1;
          const p = pointAlong(routeRaw, t);
          attr.setXYZ(k, p[0], p[1], p[2]);
        }
        attr.needsUpdate = true;
      }
      for (const l of hopLabels) {
        tmp.copy(l.mid).applyMatrix4(group.matrixWorld).project(camera);
        const behind = tmp.z > 1;
        l.el.classList.toggle("hidden", behind);
        if (!behind) l.el.style.transform = `translate(${((tmp.x + 1) / 2) * width}px, ${((1 - tmp.y) / 2) * height}px)`;
      }
    },
    setResolution(width, height) {
      resolution = new THREE.Vector2(width, height);
      for (const m of materials) m.resolution.copy(resolution);
    },
    dispose() {
      clear();
      glow.dispose();
      group.removeFromParent();
    },
  };
}
