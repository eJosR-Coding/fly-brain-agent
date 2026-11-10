/** three.js look parameters. The values live in style.css (:root custom properties); this just reads them. */

export interface Theme {
  background: string;
  baseHex: string;
  highlightHex: string;
  /** point size = brain max dimension / pointSizeDivisor */
  pointSizeDivisor: number;
  pointOpacity: number;
  /** highlighted points are drawn this many times larger than base points */
  highlightSizeScale: number;
  highlightOpacity: number;
  /** depth fade: far points blend into fogColor between fogNear and fogFar (multiples of the brain size) */
  fogColor: string;
  fogNear: number;
  fogFar: number;
}

const FALLBACK = {
  bg: "#000000", base: "#6b6b6b", hl: "#ffffff", div: "450", opacity: "0.6",
  hlScale: "2.5", hlOpacity: "0.9", fog: "#0a0a0a", fogNear: "1.1", fogFar: "2.3",
};

function cssVar(name: string, fallback: string): string {
  if (typeof document === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export function readTheme(): Theme {
  const hl = cssVar("--highlight", FALLBACK.hl);
  return {
    background: cssVar("--bg", FALLBACK.bg),
    baseHex: cssVar("--point-base", FALLBACK.base),
    highlightHex: hl,
    pointSizeDivisor: Number(cssVar("--point-size-divisor", FALLBACK.div)),
    pointOpacity: Number(cssVar("--point-opacity", FALLBACK.opacity)),
    highlightSizeScale: Number(cssVar("--highlight-size-scale", FALLBACK.hlScale)),
    highlightOpacity: Number(cssVar("--highlight-opacity", FALLBACK.hlOpacity)),
    fogColor: cssVar("--fog-color", FALLBACK.fog),
    fogNear: Number(cssVar("--fog-near", FALLBACK.fogNear)),
    fogFar: Number(cssVar("--fog-far", FALLBACK.fogFar)),
  };
}
