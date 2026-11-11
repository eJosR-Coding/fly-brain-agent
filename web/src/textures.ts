import * as THREE from "three";

/** Round sprite for gl_PointCoord: a crisp disc, or a disc with a soft glow falloff for highlights. */
export function dotTexture(glow: boolean): THREE.CanvasTexture {
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

/** A thin ring with a soft halo, for the hover and selection markers. */
export function ringTexture(): THREE.CanvasTexture {
  const size = 96;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d")!;
  const r = size / 2;
  const halo = g.createRadialGradient(r, r, r * 0.55, r, r, r);
  halo.addColorStop(0, "rgba(255,255,255,0.35)");
  halo.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = halo;
  g.fillRect(0, 0, size, size);
  g.strokeStyle = "#fff";
  g.lineWidth = 4;
  g.beginPath();
  g.arc(r, r, r * 0.55, 0, Math.PI * 2);
  g.stroke();
  return new THREE.CanvasTexture(c);
}
