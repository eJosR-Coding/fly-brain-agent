/** Derived state of the chips (each value is computed from its sources, never poked from several places). Pure. */
export function chartsChipState(s: { hasContent: boolean; panelOpen: boolean }): { disabled: boolean; active: boolean } {
  return { disabled: !s.hasContent, active: s.panelOpen };
}

export function simChipsDisabled(s: { simulating: boolean; streaming: boolean; simulationOff: boolean }): boolean {
  return s.simulating || s.streaming || s.simulationOff;
}
