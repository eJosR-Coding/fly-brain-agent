/** Mounts the static fly body into a host element. Failures are swallowed (a console warning): it must never block boot. */
import { createBodyScene, type BodyScene } from "./body";
import { loadBody } from "./bodyData";
import type { Theme } from "./theme";

export const BODY_CAPTION = "static model (NeuroMechFly v2 micro-CT mesh, Apache-2.0), not driven by the brain";

export interface MountedBody {
  remove(): void;
}

export interface BodyDeps {
  theme: Theme;
  load?: typeof loadBody;
  create?: typeof createBodyScene;
  reducedMotion?: boolean;
}

export function mountBody(host: HTMLElement, deps: BodyDeps): MountedBody {
  const stage = document.createElement("div");
  stage.className = "body-stage";
  const caption = document.createElement("span");
  caption.className = "body-caption";
  caption.textContent = BODY_CAPTION;
  let scene: BodyScene | null = null;
  let removed = false;
  void (async () => {
    try {
      const body = await (deps.load ?? loadBody)();
      if (removed) return;
      host.appendChild(stage);
      scene = (deps.create ?? createBodyScene)(stage, body, deps.theme, { reducedMotion: deps.reducedMotion });
      stage.appendChild(caption);
    } catch (e) {
      stage.remove();
      console.warn("fly body not shown:", e);
    }
  })();
  return {
    remove() {
      removed = true;
      scene?.dispose();
      scene = null;
      stage.remove();
    },
  };
}
