# Feature: harden T1-T4 (break it before T5)

## Objective
Try to break every layer built so far (atlas, selections/tools, MCP server, agent + CLI) with edge cases and hostile inputs, then pin each real failure with a test and fix it.

## Problem / why
Everything was built happy-path first with TDD. Before exposing it over HTTP (T5), where inputs come from browsers and users, the inner layers must fail loudly and helpfully, never crash, hang, leak, or return wrong data silently.

## Scope
- In: `flybrain.atlas`, `selections`, `tools`, `mcp_server`, `agent`, `agent_mcp`, `cli`. Adversarial exploration, then regression tests + fixes.
- Out: new features (lesion, gain, web API, frontend), performance tuning beyond obvious blowups.

## Constraints
- Failures found by exploration are evidence, not automatic scope: design-level changes go to the user first.
- Strict TDD: every fix starts with a RED test reproducing the failure.
- No network / API key in tests. uv only. Conventional commits. GitHub Flow: branch -> PR -> user merges.

## TDD
- Mode: strict (source: global user config). Runner: `uv run pytest`.

## Tasks
- [ ] H0 Break-it exploration: adversarial probes per layer, report concrete repro + observed vs expected + severity. Route: delegated explorer (read-only on repo, scratch scripts only).
- [ ] H1 Atlas + selections + tools fixes. Route: delegated writer.
- [ ] H2 MCP server fixes. Route: delegated writer.
- [ ] H3 Agent + agent_mcp + CLI fixes. Route: delegated writer.

## Acceptance criteria
- Every confirmed failure has a regression test that was RED before its fix.
- `uv run pytest` green.
- Findings that are design decisions are listed and decided with the user, not silently "fixed".

## Delivery
- Branch `test/hardening` from main; PR to main, user merges. Dependency changes (if any) in their own `chore(deps)` commit.

## Progress / evidence
- Engram mirror: pending (mem_save blocked by hook/auto-mode in this session).

## Next step
H0.
