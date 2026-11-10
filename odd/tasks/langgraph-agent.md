# Feature: LangGraph agent over the flybrain MCP server

## Objective
A provider-agnostic chat agent that answers questions about the fly brain by calling the flybrain MCP tools.

## Problem / why
The MCP server works from Claude Code, but the product needs its own agent so users can pick any model (LangChain `init_chat_model`) and so the future web API (T5) can stream the agent's steps.

## Scope
- In: `flybrain.agent` (hand-built LangGraph loop: model node -> `tools_condition` -> ToolNode -> back), tools loaded from the MCP server through `langchain-mcp-adapters` (the agent is an MCP client), hard stop via `recursion_limit` with a graceful message, a CLI `flybrain-chat --model provider:model`.
- Out: web API, streaming over HTTP, frontend, new brain tools (lesion, gain), persistence of conversations.

## Constraints
- Hexagonal: the agent depends on MCP tools, never imports `flybrain.atlas` directly.
- `mcp<2` stays: `langchain-mcp-adapters` 0.3.2 requires `mcp<2.0.0,>=1.24.0` (verified on PyPI 2026-11-09).
- Tests never call a real LLM or need an API key (fake chat model).
- Never print to stdout from the MCP server (stdio transport).
- uv only. Conventional commits, no AI attribution.

## TDD
- Mode: strict (source: global user config "Strict TDD Mode: enabled"). Runner: `uv run pytest`.

## Tasks
- [x] T1 Agent graph: `build_graph(model, tools)` with the explicit loop, system prompt, `recursion_limit` hard stop. Unit tests with a fake model that emits tool_calls against fake tools. Route: delegated writer (2+ non-trivial files).
- [x] T2 MCP wiring + CLI: load tools from `flybrain-mcp` over stdio via `langchain-mcp-adapters`; `flybrain-chat` CLI using `init_chat_model`. Integration test: fake model calls `count_neurons` on the real MCP server with a tiny annotations TSV. Route: delegated writer.

## Acceptance criteria
- `uv run pytest` green, no network, no API key.
- A fake model asking for `count_neurons` gets the server's real result and the graph ends when the model stops asking for tools.
- A model that never stops hits the hard stop and returns a clear message, not a crash.
- `flybrain-chat --help` works; with a real key, `flybrain-chat --model anthropic:claude-sonnet-5-5 "..."` answers.

## Delivery
- Strategy: ask-on-risk. Branch `feat/agent` stacked on `feat/mcp-server`. Forecast ~350 authored lines.

## Progress / evidence
- T1 RED: `tests/test_agent.py` -> `E   ModuleNotFoundError: No module named 'flybrain.agent'` (1 collection error).
- T1 GREEN: `uv run pytest -q` -> 36 passed (31 before + 5 new). Versions: langgraph 1.2.20, langchain-core 1.7.7, mcp 1.30.0 (mcp<2 kept).
- T1 commit: 3d25f1e
- Engram mirror: pending (mem_save blocked by hook/auto-mode in this session).

- Review (2026-11-10): slice 03f8f7c..0dc810d too big for the reviewer context (lens_context_budget_exceeded, 1072 lines, mostly uv.lock). Split: ticket-fix slice 03f8f7c..14c5d92 -> user declined. Agent slice 14c5d92..0dc810d (1016 lines) -> granted but lens_context_budget_exceeded again; no authority created. Unreviewed; uv.lock growth makes any slice with a dependency add exceed the budget.
- Fix (user-authorized history rewrite, 2026-11-10): regrouped commits so dependency changes live alone in 88c6894 (`chore(deps)`: pyproject + uv.lock + .gitattributes marking uv.lock generated). Generated lock is not reviewable under the lens budget; code is reviewed from base 56a6e6e (both chore(deps) commits first). Code slice 88c6894..3d25f1e: 208 lines, medium, review_due=false (under_budget) -> pending, reviewed together with T2. Old history kept on branch backup/feat-agent-before-split.
- Delivery rule from now on: any dependency add goes in its own `chore(deps)` commit, so code slices stay reviewable.

- T2 deps commit: 56a6e6e (`chore(deps)`). Versions: langchain-mcp-adapters 0.3.2, langchain 1.4.9, langchain-anthropic 1.7.10, mcp 1.30.0 (mcp<2 kept).
- T2 RED: `tests/test_cli.py:3: E   ImportError: cannot import name 'cli' from 'flybrain'` (+ `flybrain.agent_mcp` missing; 2 collection errors).
- T2 GREEN: `uv run pytest -q` -> 40 passed (36 + 4 new: 2 MCP integration, 2 CLI). `uv run flybrain-chat --help` -> exit 0. Without network/key the CLI prints a one-line friendly error, no traceback.
- T2 code commit: 28df57c (agent_mcp.py, cli.py, tests, README, `flybrain-chat` script).
- Session finding: `.venv/.../langchain_mcp_adapters/client.py` `get_tools()` docstring: "A new session will be created for each tool call" (calls `load_mcp_tools(None, connection=...)`). Verified by hand: search_neurons ticket then list_selection via get_tools() -> `unknown selection ... known selections: none yet`. Fix: `client.session("flybrain")` (client.py) kept open + `load_mcp_tools(session)` (tools.py); one server process per conversation, tickets survive. Covered by `test_ticket_survives_across_tool_calls`.
- Route: delegated writer (2+ non-trivial files).

- Review (code-only slice 56a6e6e..HEAD, 417 lines, 9 files, medium): granted, 1 lens (review-reliability), approved, acknowledged (authority burned, lineage review-421081ce0c4b408f). Second deps commit moved next to the first via user-authorized rebase so code commits are contiguous. Backups: backup/feat-agent-before-split, backup/feat-agent-before-split-2.

## Next step
Review slice 56a6e6e..HEAD (code only), then T5 FastAPI.
