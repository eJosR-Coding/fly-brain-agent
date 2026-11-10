# Feature: atlas core + MCP server

Locator: `odd/tasks/atlas-mcp.md` · Engram mirror: `odd/atlas-mcp/tasks` (project fly-brain-agent)

## Objective
First two layers of fly-brain-agent: a tested core that knows where every FlyWire neuron is and what it is, and an MCP server that exposes it as tools any agent can call.

## Why
Every later layer (LangGraph agent, FastAPI, Three.js viewer) depends on these. The MCP server can be tried in Claude Code before our own agent exists.

## Scope
- In: `flybrain.atlas` (load annotations, positions in µm, queries), MCP server with search / count / describe tools, README, tests.
- Out: connectome edges (paths between neurons), simulation, agent, API, frontend.

## Constraints
- FlyWire voxel size 4 × 4 × 40 nm: positions are converted with x·4, y·4, z·40 nm, reported in µm.
- Core is pure Python (no MCP / web imports). Tools are thin wrappers over the core.
- Tool outputs stay small (capped lists) so they fit in an LLM context.
- uv only. Conventional commits, no AI attribution.

## TDD
Mode: strict (source: global orchestrator config "Strict TDD Mode: enabled"). Runner: `uv run pytest`.

## Tasks
- [x] T1 Scaffold repo (git, uv, README, .gitignore). Route: inline (mechanical).
- [x] T2 `flybrain.atlas`: load annotations, µm positions, find / count_by / describe. Route: delegated writer (2+ non-trivial files).
- [x] T3 MCP server (FastMCP) wrapping the atlas + how to register it in Claude Code. Route: delegated writer. Design (accepted with user): tools return small summaries + a selection ticket (selection_id) instead of full row lists, so 86k-neuron results cost ~200 tokens; selections stay server-side for later actions (highlight, lesion). Branch: feat/mcp-server (stacked on feat/atlas-core).
- [x] T3a Selection ids are random (`sel_<6 hex>`), not a counter: a counter restarts at sel_1 after a server restart, so a stale ticket would silently point at other neurons. Decided with user: unique session-scoped ids (fail loudly) over persistence. Route: inline (2 small files, design settled).

## Acceptance criteria
- `uv run pytest` green; tests use a tiny fixture, not the 31 MB file.
- Real data loads: 139,248 neurons, brain extent ≈ 815 × 392 × 278 µm.
- MCP server starts and lists its tools.

## Progress / evidence
- T1: commit 2d65217 (main).
- T2 RED: `uv run pytest -q` failed at collection with `ModuleNotFoundError: No module named 'flybrain.atlas'` before implementation.
- T2 GREEN: `uv run pytest -q` -> 16 passed (15 fixture tests + 1 real-data integration). Real data: 139,248 neurons, extent within 2% of 815 x 392 x 278 um.
- T2: commit 35f2375 (feat/atlas-core). Parent re-ran pytest: 16 passed. Review assess: medium, review_due=false (under_budget, 198 lines) -> stays pending in slice. Engram mirror: pending (mem_save blocked by hook/auto-mode).
- T3 RED: `uv run pytest -q` -> 3 collection errors (no flybrain.tools / selections / mcp_server) before implementation.
- T3 GREEN: `uv run pytest -q` -> 30 passed. Dopamine search on real data: total 5909, JSON 1317 chars. `timeout 5 uv run flybrain-mcp </dev/null` -> exit 0. mcp pinned <2 (v2 renamed FastMCP to MCPServer).

- T3: commit bb33895 (feat/mcp-server). Parent re-ran pytest: 30 passed. Verified mcp 2.3.0 is latest on PyPI; pinned mcp<2 (1.30.0) because 2.x renamed FastMCP -> MCPServer. Revisit when adding LangGraph adapters.

- Review (atlas + MCP slice, base 6806fdd..7cfbd1d): granted, 1 lens (reliability), approved, acknowledged (authority burned). Next reviewed boundary: 7cfbd1d.

- T3a RED: `uv run pytest -q` -> 2 failed (`assert 'sel_1' != 'sel_1'` on the restart test), 29 passed.
- T3a GREEN: `uv run pytest -q` -> 31 passed. Commit 1767b1d. Review assess: medium, review_due=false (under_budget) -> pending in slice.

## Next step
T4 LangGraph agent.
