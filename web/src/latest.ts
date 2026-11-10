/** "Latest wins": only the most recently started request may apply its result. */

export class Latest {
  private n = 0;

  /** Start a request; returns its token. */
  next(): number {
    return ++this.n;
  }

  isCurrent(token: number): boolean {
    return token === this.n;
  }

  /** Make every in-flight request stale (e.g. the clear chip). */
  invalidate(): void {
    this.n++;
  }
}

/**
 * Run `work`, then `apply` its result only if no newer request started meanwhile.
 * Resolves true when applied. A stale failure is swallowed (nobody is waiting on it);
 * a failure of the current request is rethrown.
 */
export async function runLatest<T>(
  latest: Latest,
  work: () => Promise<T>,
  apply: (result: T) => void,
): Promise<boolean> {
  const token = latest.next();
  let result: T;
  try {
    result = await work();
  } catch (e) {
    if (latest.isCurrent(token)) throw e;
    return false;
  }
  if (!latest.isCurrent(token)) return false;
  apply(result);
  return true;
}
