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
- [x] H1 Atlas + selections + tools fixes. Route: delegated writer.
- [x] H2 MCP server fixes. Route: delegated writer.
- [x] H3 Agent + agent_mcp + CLI fixes. Route: delegated writer.

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
- H0-15 cosmetic MCP stdin noise (ACCEPTED, left as is by user decision).
Design decisions (asked to user):
- H0-02 empty-string / unstripped filters; H0-08 dead server mid-session; H0-11 silent clamping; H0-12 no-match hints without valid values (+ side docstring missing center/na); H0-14 tickets never evicted (~2.2 MB per whole-brain ticket).
Probe caveat: one CLI probe reached the real init_chat_model (anthropic, no key) and got "Connection error"; no .env read, no answer returned.

## Accepted decisions (user, applied in H1-H3)
- H0-02: empty/whitespace filter = no filter for every field; values stripped; >200 chars rejected (not truncated); non-str -> TypeError.
- H0-08: dead MCP server mid-run stops `ask` with `ServerStoppedError("the flybrain server stopped: <cause>")`; detected from transport errors (McpError code -32000, anyio Closed/BrokenResourceError, EndOfStream, BrokenPipe), not from the generic ToolMessage. Verified against a real dying server (probe: first call McpError "Connection closed", later calls ClosedResourceError).
- H0-11: out-of-range limit/offset/top are clamped, never an error, and a `note` field says so. Caps: page 50, count_neurons top 100.
- H0-12: no-match hint lists valid values for nt / super_class / side computed from the atlas. Real `side` values: left, right, center, na (docs fixed).
- H0-14: SelectionStore LRU cap (default 100, env FLYBRAIN_MAX_SELECTIONS, bad value = clear error), add and get refresh recency, Selection stores `filters`, evicted ticket error names the search to re-run (tombstones, max 1000), unknown-ticket error shows at most 10 recent ids + "and N more".
- H0-15: accepted as is.

## Decided differently / notes
- H0-06 budget: `--max-steps` now means tool rounds. recursion_limit alone cannot do it exactly (langgraph needs 2n+2 for n rounds + answer, and a looping model then runs n+1 tools), so the router counts rounds from the message history (config `max_tool_rounds`) and recursion_limit=2n+2 is only a backstop. Default MAX_STEPS changed 12 supersteps -> 6 tool rounds (same real budget).
- H0-07: a message with BOTH valid and invalid tool calls routes to the tools node only; the invalid ones are dropped (provider-specific, left as is).
- H0-05(d): the server's real cause goes to the child's stderr (already printed above our line); our line says "flybrain server failed to start: <client-side cause> (see the server message above, if any)" because the MCP client only sees "Connection closed".
- Invalid FLYBRAIN_MAX_SELECTIONS is reported by flybrain-mcp like a bad annotations file (one stderr line, exit 1).

## Progress / evidence
- Engram mirror: pending (mem_save blocked by hook/auto-mode in this session).
- Strict TDD, runner `uv run pytest`. Baseline 40 pass. Final: 155 pass.
- H1 (delegated writer) commits: 7711e61 fix(atlas), 78551f3 feat(selections), 41dedf6 fix(tools).
  - atlas RED: `ImportError: cannot import name 'AtlasLoadError' from 'flybrain.atlas'`; with the two exception classes stubbed, 35 failed / 16 passed (negative/non-int limit, filters, load errors, describe ids, numeric cell_type). GREEN 51 pass in file; suite 75.
  - selections RED: 12 failed / 4 passed (no cap, no `filters`, no expiry message, uncapped known list). GREEN: suite 87.
  - tools RED: 18 failed / 16 passed (echo of empty filters, no `note`, generic hint, no type checks, `DID NOT RAISE` on float ids). GREEN: suite 112. Two of my own expectations were wrong (62 vs 61 count, id 12 exists in the fixture) and were fixed in the test, not the code.
- H2 commit: 0069cd0 fix(mcp). RED: 5 failed / 3 passed (`AtlasLoadError` traceback instead of one line, `ValueError: FLYBRAIN_MAX_SELECTIONS ...` uncaught, subprocess stderr had 11 lines, side docstring lacked "center"). The unknown-neuron MCP text test was already green after the atlas fix. GREEN: suite 118.
- H3 commits: c2fce5c fix(agent), 5baaf33 fix(cli). 
  - agent RED (after stubbing ServerStoppedError): 19 failed / 6 passed (max_steps validation, 1 round + answer, malformed calls, empty reply, 5 dead-server variants). GREEN: suite 138.
  - cli RED: 13 failed / 6 passed (e.g. `could not use model 'anthropic:claude-sonnet-5-5': Connection closed` for a server failure, no argparse validation). GREEN: suite 155.
- Verification: `uv run pytest -q` 155 passed; `uv run flybrain-chat --help` ok (mentions tool rounds); `FLYBRAIN_ANNOTATIONS=/nonexistent uv run flybrain-mcp </dev/null` -> stderr `flybrain-mcp: annotations file not found: /nonexistent`, exit=1, empty stdout. Real-server probe: fake model calling a tool on a server that exits -> `ServerStoppedError: the flybrain server stopped: Connection closed`. No real LLM or network was used.
- README updated (docs(readme) commit).

- Review (2026-11-10), 4 slices (each under the reviewer budget), all granted, approved, acknowledged (authority burned):
  A main..78551f3 atlas+selections (448 lines, medium, 1 lens) review-143319ddbe7f86bd;
  B 78551f3..0069cd0 tools+mcp (305 lines, HIGH, 4 lenses) review-731fefe83cdd771f;
  C 0069cd0..c2fce5c agent (298 lines, medium, 1 lens) review-5880cab9e3da5f91;
  D c2fce5c..a531b1d cli+docs (267 lines, medium, 1 lens) review-fd38f8efceb99ec7.
- Non-blocking advisory follow-ups (not done): numpy ints rejected by describe (atlas.py:98); root_id upper bound (atlas.py:100); store bounds tests (selections.py:53); clamp-note asymmetry and duplicated note assembly (tools.py:23,127); doc constants duplicated (mcp_server.py:34); describe coercion (tools.py:104); mixed valid/invalid tool calls drop invalid silently (agent.py:55); transport error classes untested (agent_mcp.py:52); ExceptionGroup unwrap + KeyboardInterrupt/ServerStopped inside groups in CLI (cli.py:92-107).

## Next step
Open PR test/hardening -> main for the user to review and merge; then T5 (HTTP).
