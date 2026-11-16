/** Ordering of the cables against the highlight that precedes them and against anything that replaces them.
 *  A path request is valid only while nothing newer (another path, a highlight, a recording, a clear) came. */
export class PathOrder {
  private token = 0;
  private highlightApplied = true;

  /** The chat's highlight finished: `applied` is false when a newer selection superseded it. */
  highlightSettled(applied: boolean): void {
    this.highlightApplied = applied;
  }

  /** Start a path request; null when its own highlight was superseded (nothing to draw, nothing cancelled). */
  begin(): number | null {
    const applied = this.highlightApplied;
    this.highlightApplied = true; // the verdict belongs to one path only
    return applied ? ++this.token : null;
  }

  isCurrent(token: number): boolean {
    return token === this.token;
  }

  /** Something replaced or cleared the cables: a path still waiting for the scene is dropped. */
  cancel(): void {
    this.token++;
  }
}
