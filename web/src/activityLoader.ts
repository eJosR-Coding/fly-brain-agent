/** Latest-wins loading of recordings: a slow older load never replaces a newer one. */
import { runLatest, type Latest } from "./latest";

/** Run `load`, then `show` its result only if no newer recording was requested meanwhile (true when shown). */
export function loadLatestActivity<T>(latest: Latest, load: () => Promise<T>, show: (loaded: T) => void | Promise<void>): Promise<boolean> {
  return runLatest(latest, load, (loaded) => void show(loaded));
}
