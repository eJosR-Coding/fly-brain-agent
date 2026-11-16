/** Atomic scene bring-up: create, wire, and only then publish. A wiring failure rolls everything back (undo the
 *  wiring in reverse, dispose the scene) and rethrows, so a retry builds exactly one scene. Pure (no DOM). */
export function commitScene<S extends { dispose(): void }>(o: {
  create: () => S;
  /** wire the scene to the page; register every side effect that needs undoing through `undo` */
  wire: (scene: S, undo: (fn: () => void) => void) => void;
  /** the point of no return: only plain assignments and removals of the placeholders */
  publish: (scene: S) => void;
}): S {
  const scene = o.create();
  const undos: (() => void)[] = [];
  try {
    o.wire(scene, (fn) => void undos.push(fn));
  } catch (e) {
    for (const fn of undos.reverse()) {
      try {
        fn();
      } catch {
        /* a failing undo must not hide the original error or stop the rest of the rollback */
      }
    }
    scene.dispose();
    throw e;
  }
  o.publish(scene);
  return scene;
}
