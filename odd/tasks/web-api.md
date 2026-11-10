# Feature: T5 web API (FastAPI) for the brain viewer

## Objective
An HTTP backend for the future Three.js viewer: brain points in binary, and a chat endpoint that streams the agent's steps (SSE), including which neurons to highlight.

## Problem / why
The agent only runs as a CLI. A browser needs (1) all 139,248 positions once, in a GPU-ready format, and (2) live events while the agent works, with highlights as row indices (the browser has no root_ids).

## Design (agreed with user, see notes 14 and 16)
- `GET /brain/meta` -> JSON contract: count, dtype float32, little-endian, layout xyz interleaved, row order = atlas row order, extent in um.
- `GET /brain/points` -> `application/octet-stream`, float32 LE, xyz interleaved, atlas row order (~1.67 MB).
- `POST /chat` {message, conversation_id?} -> `text/event-stream` (SSE). Events: `tool_call`, `highlight` {selection_id, total, indices}, `answer`, `error`, `done`. Multi-turn memory per conversation_id (LangGraph checkpointer).
- `GET /selections/{id}/indices` -> binary uint32 LE row indices (for huge selections).
- The web app hosts the MCP server IN-PROCESS (in-memory transport) so the atlas loads once and the API can map a ticket's root_ids to row indices from the same SelectionStore. The agent still talks to tools only over MCP (hexagonal).
- SSE, not WebSocket: the client sends one message, the server streams back; WebSocket only if a live two-way control (e.g. a dopamine slider during simulation) appears later.

## Scope
- In: `flybrain.api` (FastAPI app factory + lifespan), in-memory MCP wiring, SSE event mapping, `flybrain-web` script (uvicorn), CORS for the Vite dev origin, tests.
- Out: frontend (T6), lesion/gain tools, auth, deployment.

## Constraints
- Hexagonal: api is an adapter; core untouched except tiny helpers if needed (e.g. row-index lookup).
- Hardened from day one: message length limit, empty message 422, model/server failures become an `error` event then `done`, never a hung stream; bounded event sizes.
- Tests: no network, no API key (fake chat model), tiny TSV fixture.
- Dependencies in their own `chore(deps)` commit. GitHub Flow: PR, Claude merges after checks.

## TDD
- Mode: strict (source: global user config). Runner: `uv run pytest`.

## Tasks
- [ ] W1 Brain data endpoints: /health, /brain/meta, /brain/points, row-index lookup. Route: delegated writer.
- [ ] W2 Chat SSE: in-process MCP, agent streaming to events, highlight indices, conversation memory, /selections/{id}/indices, error paths, `flybrain-web`. Route: delegated writer.

## Acceptance criteria
- Points endpoint returns exactly count*12 bytes; decoding with numpy `<f4` matches the atlas.
- A fake model calling search_neurons produces tool_call -> highlight (correct indices) -> answer -> done, in order.
- Failures produce an `error` event and close the stream.
- `uv run pytest` green.

## Progress / evidence
- Engram mirror: pending (mem_save blocked by hook/auto-mode in this session).

## Next step
W1.
