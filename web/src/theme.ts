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
  /** depth cue: brightness gain and size multiplier for the farthest and the nearest points of the cloud */
  depthGainFar: number;
  depthGainNear: number;
  depthSizeFar: number;
  depthSizeNear: number;
  /** auto-orbit speed (OrbitControls units; 2 = one turn per 30 s) and the pause after the user lets go (ms) */
  orbitSpeed: number;
  orbitResumeMs: number;
  /** base point tone per region name (central, optic, sensory, other) */
  tones: Record<string, string>;
}

const FALLBACK = {
  bg: "#000000", base: "#6b6b6b", hl: "#ffffff", div: "450", opacity: "0.6",
  hlScale: "2.5", hlOpacity: "0.9", depthGainFar: "0.4", depthGainNear: "1.3", depthSizeFar: "0.8", depthSizeNear: "1.3", orbitSpeed: "0.6", orbitResumeMs: "4000",
  toneCentral: "#d6d6d6", toneOptic: "#8c8c8c", toneSensory: "#a9bccd", toneOther: "#5e5e5e",
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
    depthGainFar: Number(cssVar("--depth-gain-far", FALLBACK.depthGainFar)),
    depthGainNear: Number(cssVar("--depth-gain-near", FALLBACK.depthGainNear)),
    depthSizeFar: Number(cssVar("--depth-size-far", FALLBACK.depthSizeFar)),
    depthSizeNear: Number(cssVar("--depth-size-near", FALLBACK.depthSizeNear)),
    orbitSpeed: Number(cssVar("--orbit-speed", FALLBACK.orbitSpeed)),
    orbitResumeMs: Number(cssVar("--orbit-resume-ms", FALLBACK.orbitResumeMs)),
    tones: {
      central: cssVar("--tone-central", FALLBACK.toneCentral),
      optic: cssVar("--tone-optic", FALLBACK.toneOptic),
      sensory: cssVar("--tone-sensory", FALLBACK.toneSensory),
      other: cssVar("--tone-other", FALLBACK.toneOther),
    },
  };
}
