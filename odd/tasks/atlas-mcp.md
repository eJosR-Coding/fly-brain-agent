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
- [ ] T3 MCP server (FastMCP) wrapping the atlas + how to register it in Claude Code. Route: delegated writer.

## Acceptance criteria
- `uv run pytest` green; tests use a tiny fixture, not the 31 MB file.
- Real data loads: 139,248 neurons, brain extent ≈ 815 × 392 × 278 µm.
- MCP server starts and lists its tools.

## Progress / evidence
- T1: commit d3a9ff1 (main).
- T2 RED: `uv run pytest -q` failed at collection with `ModuleNotFoundError: No module named 'flybrain.atlas'` before implementation.
- T2 GREEN: `uv run pytest -q` -> 16 passed (15 fixture tests + 1 real-data integration). Real data: 139,248 neurons, extent within 2% of 815 x 392 x 278 um.
- T2: commit 774b938 (feat/atlas-core). Parent re-ran pytest: 16 passed. Review assess: medium, review_due=false (under_budget, 198 lines) -> stays pending in slice. Engram mirror: pending (mem_save blocked by hook/auto-mode).

## Next step
T3.
