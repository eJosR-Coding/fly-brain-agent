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
- [x] H0 Break-it exploration: adversarial probes per layer, report concrete repro + observed vs expected + severity. Route: delegated explorer (read-only on repo, scratch scripts only).
- [ ] H1 Atlas + selections + tools fixes. Route: delegated writer.
- [ ] H2 MCP server fixes. Route: delegated writer.
- [ ] H3 Agent + agent_mcp + CLI fixes. Route: delegated writer.

## Acceptance criteria
- Every confirmed failure has a regression test that was RED before its fix.
- `uv run pytest` green.
- Findings that are design decisions are listed and decided with the user, not silently "fixed".

## Delivery
- Branch `test/hardening` from main; PR to main, user merges. Dependency changes (if any) in their own `chore(deps)` commit.

## H0 findings (2026-11-10, probes in session scratchpad h0/)
Plain bugs (fix with TDD):
- H0-01 atlas.find negative limit = head(-N) (drops last N silently); non-int limit raw pandas TypeError.
- H0-03 describe_neuron unknown id -> error text is just "12" (str(KeyError)).
- H0-04 bad/missing annotations -> pandas traceback at server start; client sees EOF; CLI blames the model.
- H0-05 CLI reports every failure as "could not use model"; no validation of --max-steps / provider:model shape.
- H0-06 ask(max_steps<=0) raises ValueError; budget counts supersteps (1 tool call + answer needs 4).
- H0-07 malformed tool call (invalid_tool_calls) or empty final content -> ask returns '' silently.
- H0-09 unbounded output: echoed filters (2M in -> 2M out), count_neurons top uncapped (120k chars), known-selections list uncapped.
- H0-10 tools.* lack own arg validation (raw TypeError/ValueError on direct calls).
- H0-13 float ids lose precision; NaN positions -> invalid JSON; non-str filters AttributeError; duplicate root_ids undetected.
- H0-15 cosmetic MCP stdin noise (left as is).
Design decisions (asked to user):
- H0-02 empty-string / unstripped filters; H0-08 dead server mid-session; H0-11 silent clamping; H0-12 no-match hints without valid values (+ side docstring missing center/na); H0-14 tickets never evicted (~2.2 MB per whole-brain ticket).
Probe caveat: one CLI probe reached the real init_chat_model (anthropic, no key) and got "Connection error"; no .env read, no answer returned.

## Progress / evidence
- Engram mirror: pending (mem_save blocked by hook/auto-mode in this session).

## Next step
H0.
