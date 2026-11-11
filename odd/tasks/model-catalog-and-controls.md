# Model catalog and grouped viewer controls

## Objective
Offer only the chat models that work for the keys a user actually has, and make the viewer's
control rows readable at a glance.

## Problem
- The model list is hardcoded (`DEFAULT_EXTRA_MODELS`, `--models`); a retired id (Llama on Groq)
  stayed in the picker and failed at chat time. People bring different keys.
- 17 identical chips mix four jobs (view, run an experiment, highlight, clear) with no grouping;
  the region legend competes with them.

## Scope
- T1 Hybrid catalog: curated tool-calling models per provider, shown only when that provider's key
  is set, cross-checked against the provider's `/models` list at startup (retired ids hidden, fail
  open on network errors). Default model = first available in catalog order (cheapest first).
  `FLYBRAIN_MODELS` / `--models` still override.
- T2 Grouped controls: labelled VIEW / RUN / HIGHLIGHT groups, experiments marked as slow, clear
  set apart; tidy the region legend. Strict black/white design, 12px mono minimum, contrast >= 4.5:1.

## Constraints
- Key values are sent only to their own provider (as chat already does), never logged or returned.
- Tests never hit the network (inject the fetcher).
- No invented model ids: only ids verified earlier in this project.
- Deps in their own `chore(deps)` commit.

## Tasks
- [x] T1 hybrid model catalog (route: delegated writer, 3+ files: models.py, web.py, tests)
  Evidence: 322d756, 579 passed (uv run pytest -q), RED seen first (import errors, 4 failing web/api tests).
- [ ] T2 grouped viewer controls (route: delegated writer, main.ts/style.css/index.html + tests)

## Checks
- TDD strict (project/session config). Runners: `uv run pytest -q`, `pnpm -C web test`, `pnpm -C web build`.
- T2: screenshots at 1440x900 and 1366x768.

## Delivery
Strategy: ask-on-risk. Forecast ~350 authored lines; one PR per task.

## Progress
- Created 2026-11-11.
