/** three.js look parameters. The values live in style.css (:root custom properties); this just reads them. */
import * as THREE from "three";
import type { RGB } from "./colors";

export interface Theme {
  background: string;
  baseRgb: RGB;
  highlightHex: string;
  highlightRgb: RGB;
  /** point size = brain max dimension / pointSizeDivisor */
  pointSizeDivisor: number;
  pointOpacity: number;
}

const FALLBACK = { bg: "#000000", base: "#6b6b6b", hl: "#ffffff", div: "450", opacity: "0.6" };

function cssVar(name: string, fallback: string): string {
  if (typeof document === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function rgb(hex: string): RGB {
  const c = new THREE.Color(hex);
  return [c.r, c.g, c.b];
}

export function readTheme(): Theme {
  const hl = cssVar("--highlight", FALLBACK.hl);
  return {
    background: cssVar("--bg", FALLBACK.bg),
    baseRgb: rgb(cssVar("--point-base", FALLBACK.base)),
    highlightHex: hl,
    highlightRgb: rgb(hl),
    pointSizeDivisor: Number(cssVar("--point-size-divisor", FALLBACK.div)),
    pointOpacity: Number(cssVar("--point-opacity", FALLBACK.opacity)),
  };
}
