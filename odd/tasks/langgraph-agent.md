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
- [ ] T2 MCP wiring + CLI: load tools from `flybrain-mcp` over stdio via `langchain-mcp-adapters`; `flybrain-chat` CLI using `init_chat_model`. Integration test: fake model calls `count_neurons` on the real MCP server with a tiny annotations TSV. Route: delegated writer.

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
- T1 commit: cb730db
- Engram mirror: pending (mem_save blocked by hook/auto-mode in this session).

- Review (2026-11-10): slice 03f8f7c..0dc810d too big for the reviewer context (lens_context_budget_exceeded, 1072 lines, mostly uv.lock). Split: ticket-fix slice 03f8f7c..14c5d92 -> user declined. Agent slice 14c5d92..0dc810d (1016 lines) -> granted but lens_context_budget_exceeded again; no authority created. Unreviewed; uv.lock growth makes any slice with a dependency add exceed the budget.

## Next step
T2.
