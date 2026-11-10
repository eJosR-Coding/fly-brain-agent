/** Auto-orbit gate: spin only while enabled and idle. Pure, driven by explicit timestamps (ms). */

/** Reduced-motion users start with the orbit off. */
export const orbitDefault = (prefersReducedMotion: boolean): boolean => !prefersReducedMotion;

export interface OrbitGate {
  enabled: boolean;
  setEnabled(on: boolean): void;
  /** Pointer went down on the canvas: pause. */
  press(): void;
  /** Pointer released at `now`: resume after the delay. */
  release(now: number): void;
  /** Any other camera change (preset, wheel) at `now`: pause for the delay. */
  touch(now: number): void;
  /** Whether the camera should auto-rotate at `now`. */
  active(now: number): boolean;
}

export function createOrbitGate(enabled: boolean, resumeMs: number): OrbitGate {
  let down = false;
  let resumeAt = -Infinity;
  return {
    get enabled() { return enabled; },
    set enabled(v: boolean) { enabled = v; },
    setEnabled(on) { enabled = on; },
    press() { down = true; },
    release(now) { down = false; resumeAt = now + resumeMs; },
    touch(now) { resumeAt = Math.max(resumeAt, now + resumeMs); },
    active(now) { return enabled && !down && now >= resumeAt; },
  };
}
