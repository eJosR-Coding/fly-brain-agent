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
- [x] S1 P1 `loadScene` atomic: build everything, publish `brainScene`/remove preview and open control LAST, `scene.dispose()` in the catch.
- [x] S2 P2 persisted `layout.main === "brain"` must open the brain at boot (today only on a swap).
- [x] S3 P2 `highlight()`/`showActivity` bump `pathToken`; `onPath` is skipped when its highlight was superseded (main-only flag).
- [x] S4 P2 one owner for the legend/stage: small `Stage` union and one `paintLegend` (no overwriting a highlight legend by a wave, `onNewChat` clears only a path legend, the pending timer restores).
- [x] S5 P3 `syncChartsChip()` derives disabled/active from `shown`, charts content and `charts.open`.
- [x] S6 P3 `renderSimChips()` derives disabled from `simulating || streaming || health`.
- [x] S7 P3 rename the shadowed `const shown`; hide/show of the inset must not re-trigger `openBrain` unless `main` changed.
### B. web chat and charts (web/src/chat.ts, chartsPanel.ts)
- [x] S8 P2 turn lifecycle: re-enable input in `finally`/top of `onDone`, per-turn `AbortController` with a stop affordance, ignore stream events after `done`.
- [x] S9 P2 chart playheads re-synced after a redraw (remember the last player time).
- [x] S10 P3 status line shows the model the running turn started with.
### C. python API and agent (src/flybrain/api.py, agent.py)
- [x] S12 P1 per-conversation turn guard (a second concurrent turn on one conversation_id gets an error event); prune idle entries.
- [x] S13 P1 honest text for a cut tool call ("the tool may still have run") and track in-flight tool calls.
- [x] S14 P2 experiment deadline grants tracked per active call and released on result; `running_experiment` derived.
- [x] S15 P2 `/health` chat reflects a dead in-process MCP session (`tools_ok`).
### D. python simulation and tools (sim_service.py, selections.py, experiments.py, silencing.py, tools.py doll bits, simulation.py, mcp_server.py connectome loader)
- [x] S17 P2 `SelectionStore` lock around add/get/list.
- [x] S18 P2 experiment progress covers waiting/loading/baseline (open `_Progress` inside `exclusive()` before `simulation()`).
- [x] S19 P3 `stimulate` baseline/silenced pair stored under one lock (`add_pair`), docstring made honest.
- [x] S20 P3 simulator load failure recorded (`_load_error`, non-transient only), `/health` consistent.
- [x] S21 P3 `doll.last` keeps duration and trials; `change_vs_previous` says when they differ.
- [x] S22 P3 the reason a connectome was skipped reaches the tool error.
- [x] S23 P3 `SimResult.trial_counts` typed non-Optional.

## Left as is (documented design choices, not changed)
- Doll, selection tickets and simulations shared by every conversation of the server (documented; a product choice).
- Chart store (12) vs record store (24) expire independently (the viewer gets an honest 404 "expired"); documented in the sim_service docstring only.
- Graph tri-state sentinel in api.py and the model-registry clocks (works; a comment is enough); dangling-call matching by id only; conversation_id after a restart.
- Thinking line after a finished tool, and other P3 low items with no failing scenario.

## Checks
TDD strict (source: project config). Runners: `uv run pytest -q` (baseline 821 on stack-15), `pnpm -C web test` (508), `pnpm -C web build`.
Delivery: stacked PRs after #49 (feat/stack-16-review-followups); one PR per writer area, chained in the order A, B, C, D by cherry-picking the
writers' commits. Engram mirror `odd/state-machines/tasks`: PENDING (mem_save blocked by a hook that rewrites its input).

## Results (2026-11-16)
All four areas implemented by parallel writers in isolated worktrees, chained as stacked PRs after the follow-ups PR:
- #49 follow-ups (835 py / 520 web), #50 area A main.ts (835 / 557), #51 area B chat and charts (835 / 586),
  #52 area C API and agent (853 / 586), #53 area D simulation and tools (875 / 586). Every slice passes on its own
  (pytest, web tests, build re-run by the parent). One conflict while chaining (the import line of tools.py, resolved
  keeping both sides; the test sums match: 835 + 18 + 22 = 875).
- Native reviews (consent granted each time), all APPROVED. Non-blocking findings left OPEN:
  - #49: fly_body_build.py:417-421 (WARNING) and body.ts:98-101 (suggestion).
  - #50: web/src/pathOrder.ts:13-17 (WARNING): a superseded-highlight verdict stays set until the next `begin()`; only a path
    without a preceding chat highlight could be wrongly skipped.
  - #51: chat.ts:604 (suggestion): an error after a stop can be shown as "stopped".
  - #52: none.
  - #53: tests/test_mcp_server.py:225-231 (readability WARNING) and :229 (the connectome-reason clear is not proven by a test).
- Judgment calls by the writers: the send button is the stop button while a turn runs (no automatic stall timeout, to avoid false
  timeouts on long tools); sim chips are disabled while the chat streams (behavior change); the experiment deadline is
  `base + active grants + time finished granted calls took` (the literal `base + sum(active)` made the final answer time out);
  the per-conversation guard is a set, not a Lock, because nothing waits; `set_total` in experiment progress is silent.
- NOT verified: any real browser behavior (stop on a real stalled stream, the two-canvas retry scenario, legend flows),
  real client disconnects, real provider handling of the repaired history, a real 1.5 GB simulator load.

## Progress
- 2026-11-16 done and reviewed. Next: the open findings above, and the fly-body animation plan.
