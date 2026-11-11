# Friendly results, tool cache and on-demand rendering

## Objective
Anyone can understand what a neuron or a result means without losing the technical data; repeated
experiments return instantly; the viewer stops using the GPU while nothing changes.

## Problem
- The inspector card and answers show correct but opaque data ("CB0328 · central · glutamate ·
  1,020 in"). FlyWire has 8,846 cell types but only 49 `cell_class` values (105 sub-classes), so a
  short class glossary plus a data-driven template covers every neuron.
- Simulations are deterministic (fixed seed) but always recomputed (~12 s each).
- The render loop draws every frame even when the scene is still (web/src/scene.ts:370).

## Tasks
- [x] T1 plain-language layer: class glossary (cell_class/super_class/nt) + template for unclassed
      neurons, served by the API and `describe_neuron`; one plain line atop the inspector card;
      plain captions under charts; the agent ends results with a one-line plain summary.
      (route: delegated writer, 4+ files)
- [x] T2 simulation result cache keyed by resolved row sets + parameters + seed (not ticket ids).
      (route: delegated writer)
- [x] T3 render on demand: draw only when the camera, animation, highlight or data changes.
      (route: delegated writer)
- [x] T4 small fixes from reviews: vacuous clip test (tests/test_dose_response.py:158), duplicated
      top_non_sensory helpers (silencing.py/tools.py), toolTrace DOM test (chat.ts:537), SELECTION
      line on phones. (route: delegated writer)
- [ ] T5 Engram: find why mem_save is blocked in auto mode (a hook rewrites its input); report, do
      not change user config without asking. (route: inline, read-only)

## Later (not in scope)
- Visual feedback for tools that show nothing today (part of the next experiments).

## Checks
TDD strict. Runners: `uv run pytest -q`, `pnpm -C web test`, `pnpm -C web build`; screenshots via
/tmp/claude-1000/legib/shot2.py when the UI changes.

## Delivery
ask-on-risk; one PR per task.

## Progress
- Created 2026-11-11.
- T1 done 2026-11-11 (delegated writer). Commits: glossary+backend 0fb323d, web (card line, tooltips, chart
  captions), prompt line. RED: tests/test_plain.py collection error, 10 failing web tests, prompt test.
  GREEN: `uv run pytest -q` 642 passed; `pnpm -C web test` 398 passed; `pnpm -C web build` ok.
  Screenshot /tmp/claude-1000/legib/plain.png: plain line renders under the title (13px Inter, #fff on #050505).
- T2 done 2026-11-11 (delegated writer, branch feat/sim-cache). Commit 215c064. RED: tests/test_sim_cache.py 15 failed
  (no SimulationService.run). GREEN: `uv run pytest -q` 667 passed (25 new). Cache: LRU 32 entries + 300 MB byte
  budget on SimulationService.run; output gets `cached` (stimulate) / `cached_runs` (dose_response, screen).
- T3 done 2026-11-11 (delegated writer, branch perf/render-on-demand). Commit 8f66855. RED: renderScheduler test suite
  failed (module missing). GREEN: `pnpm -C web test` 406 passed (8 new); `pnpm -C web build` ok. Playwright (swiftshader,
  own ports): idle rAF callbacks in 2 s with orbit off: 31 before -> 0 after (orbit on: ~32 both, by design); highlight
  after idle redraws then settles to 0; inspector card still follows a camera drag. Screenshots rod_*_highlight.png match.
- T4 done 2026-11-11 (delegated writer, branch fix/review-followups). Commits 4a152df (clip test + shared non_sensory_rows), dc32bf9
  (toolTrace DOM test + phone stage). RED: clip test fails with the clip removed (ci_low negative), toolTrace tests fail when
  the default is forced open. GREEN: `uv run pytest -q` 653 passed; `pnpm -C web test` 401 passed; `pnpm -C web build` ok.
  Screenshots m390_after_844.png / _667.png: ~60 px / ~28 px clearance between brain and SELECTION line.
