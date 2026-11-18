/** Mounts the static fly body into a host element. Failures are swallowed (a console warning): it must never block boot. */
import type { WorldPose } from "./bodyClip";
import { createBodyScene, type BodyScene } from "./body";
import { loadBody } from "./bodyData";
import { DEFAULT_BODY_CAPTION } from "./bodyAnimation";
import type { JointId } from "./bodyPose";
import type { Theme } from "./theme";

export const BODY_CAPTION = DEFAULT_BODY_CAPTION;

export interface MountedBody {
  remove(): void;
  /** Joint angles in degrees; joints not listed return to rest. Safe before the scene exists (the latest pose is applied on creation); a no-op after remove(). */
  setPose(angles: Partial<Record<JointId, number>>): void;
  /** One frame of a physics clip (per leg body matrices from bodyClip), or null for rest. Safe before the scene exists (the latest frame is applied on creation); a no-op after remove(). */
  setClipFrame(mats: ReadonlyMap<string, ArrayLike<number>> | null): void;
  /** The whole fly's motion in the world (root transform, floor, camera follow), or null for none. Safe before the scene exists (the latest is applied on creation); a no-op after remove(). */
  setWorld(world: WorldPose | null): void;
  /** The glass-cube sandbox on or off. Safe before the scene exists (the latest is applied on creation); a no-op after remove(). */
  setSandbox(on: boolean): void;
  /** The sandbox's objects layer (see BodyScene.setObjects); the latest list is applied on creation. Returns the validation errors once the scene exists, else none. */
  setObjects(objects: unknown): { errors: string[] };
  /** The line under the body. Safe before the scene exists (the latest text is shown on creation); a no-op after remove(). */
  setCaption(text: string): void;
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
  let pose: Partial<Record<JointId, number>> | null = null;
  let clipFrame: ReadonlyMap<string, ArrayLike<number>> | null = null;
  let world: WorldPose | null = null;
  let sandbox = false;
  let objects: unknown = [];
  let objectsSet = false;
  void (async () => {
    try {
      const body = await (deps.load ?? loadBody)();
      if (removed) return;
      host.appendChild(stage);
      scene = (deps.create ?? createBodyScene)(stage, body, deps.theme, { reducedMotion: deps.reducedMotion });
      stage.appendChild(caption);
      if (pose) scene.setPose(pose);
      if (clipFrame) scene.setClipFrame(clipFrame);
      if (world) scene.setWorld(world);
      if (sandbox) scene.setSandbox(true);
      if (objectsSet) scene.setObjects(objects);
    } catch (e) {
      stage.remove();
      console.warn("fly body not shown:", e);
    }
  })();
  return {
    setCaption(text) {
      if (removed) return;
      caption.textContent = text;
    },
    setPose(angles) {
      if (removed) return;
      pose = angles;
      scene?.setPose(angles);
    },
    setClipFrame(mats) {
      if (removed) return;
      clipFrame = mats;
      scene?.setClipFrame(mats);
    },
    setWorld(w) {
      if (removed) return;
      world = w;
      scene?.setWorld(w);
    },
    setSandbox(on) {
      if (removed) return;
      sandbox = on;
      scene?.setSandbox(on);
    },
    setObjects(list) {
      if (removed) return { errors: [] };
      objects = list;
      objectsSet = true;
      return scene?.setObjects(list) ?? { errors: [] };
    },
    remove() {
      removed = true;
      sandbox = false;
      objects = [];
      objectsSet = false;
      world = null;
      pose = null;
      clipFrame = null;
      scene?.dispose();
      scene = null;
      stage.remove();
    },
  };
}
