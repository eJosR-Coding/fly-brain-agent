# State-machine fixes from the implicit state machine audit

## Objective
Fix the implicit-state-machine problems found by a four-part read-only audit (2026-11-16): make the states explicit where
they were implicit, close the invalid combinations, and add regression tests for valid and invalid transitions.

## Authorized scope
The user approved fixing all findings on 2026-11-16 ("arreglalos todos"). Where the auditors said "leave as is" for a
documented design choice, the item is NOT changed and is listed below under "Left as is".

## Constraints (from the audit brief)
- No XState or other library; no centralizing of independent state; preserve behavior and public interfaces; no
  needless abstractions. Prefer discriminated unions, a small dict, a flag or a lock, the cheapest option first.
- Every fix gets tests for the valid transitions and for the invalid combination it closes.
- Verification evidence of the audit: the findings S1, S2 (web), S12 (no per-conversation lock in api.py), S13 (the
  "cancelled ... before this tool ran" text) and S17 (no lock in SelectionStore) were re-checked by the parent against
  the code; the rest are the auditors' reading (nothing was run).

## Tasks (4 writers in parallel, disjoint files; each in its own worktree)
### A. web/src/main.ts (+ main-only logic)
- [ ] S1 P1 `loadScene` atomic: build everything, publish `brainScene`/remove preview and open control LAST, `scene.dispose()` in the catch.
- [ ] S2 P2 persisted `layout.main === "brain"` must open the brain at boot (today only on a swap).
- [ ] S3 P2 `highlight()`/`showActivity` bump `pathToken`; `onPath` is skipped when its highlight was superseded (main-only flag).
- [ ] S4 P2 one owner for the legend/stage: small `Stage` union and one `paintLegend` (no overwriting a highlight legend by a wave, `onNewChat` clears only a path legend, the pending timer restores).
- [ ] S5 P3 `syncChartsChip()` derives disabled/active from `shown`, charts content and `charts.open`.
- [ ] S6 P3 `renderSimChips()` derives disabled from `simulating || streaming || health`.
- [ ] S7 P3 rename the shadowed `const shown`; hide/show of the inset must not re-trigger `openBrain` unless `main` changed.
### B. web chat and charts (web/src/chat.ts, chartsPanel.ts)
- [ ] S8 P2 turn lifecycle: re-enable input in `finally`/top of `onDone`, per-turn `AbortController` with a stop affordance, ignore stream events after `done`.
- [ ] S9 P2 chart playheads re-synced after a redraw (remember the last player time).
- [ ] S10 P3 status line shows the model the running turn started with.
### C. python API and agent (src/flybrain/api.py, agent.py)
- [ ] S12 P1 per-conversation turn guard (a second concurrent turn on one conversation_id gets an error event); prune idle entries.
- [ ] S13 P1 honest text for a cut tool call ("the tool may still have run") and track in-flight tool calls.
- [ ] S14 P2 experiment deadline grants tracked per active call and released on result; `running_experiment` derived.
- [ ] S15 P2 `/health` chat reflects a dead in-process MCP session (`tools_ok`).
### D. python simulation and tools (sim_service.py, selections.py, experiments.py, silencing.py, tools.py doll bits, simulation.py, mcp_server.py connectome loader)
- [ ] S17 P2 `SelectionStore` lock around add/get/list.
- [ ] S18 P2 experiment progress covers waiting/loading/baseline (open `_Progress` inside `exclusive()` before `simulation()`).
- [ ] S19 P3 `stimulate` baseline/silenced pair stored under one lock (`add_pair`), docstring made honest.
- [ ] S20 P3 simulator load failure recorded (`_load_error`, non-transient only), `/health` consistent.
- [ ] S21 P3 `doll.last` keeps duration and trials; `change_vs_previous` says when they differ.
- [ ] S22 P3 the reason a connectome was skipped reaches the tool error.
- [ ] S23 P3 `SimResult.trial_counts` typed non-Optional.

## Left as is (documented design choices, not changed)
- Doll, selection tickets and simulations shared by every conversation of the server (documented; a product choice).
- Chart store (12) vs record store (24) expire independently (the viewer gets an honest 404 "expired"); documented in the sim_service docstring only.
- Graph tri-state sentinel in api.py and the model-registry clocks (works; a comment is enough); dangling-call matching by id only; conversation_id after a restart.
- Thinking line after a finished tool, and other P3 low items with no failing scenario.

## Checks
TDD strict (source: project config). Runners: `uv run pytest -q` (baseline 821 on stack-15), `pnpm -C web test` (508), `pnpm -C web build`.
Delivery: stacked PRs after #49 (feat/stack-16-review-followups); one PR per writer area, chained in the order A, B, C, D by cherry-picking the
writers' commits. Engram mirror `odd/state-machines/tasks`: PENDING (mem_save blocked by a hook that rewrites its input).

## Progress
- 2026-11-16 created after the audit and the approval. Next step: the four writers run in parallel.
