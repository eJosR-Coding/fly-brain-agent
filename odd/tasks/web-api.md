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
- [x] W1 Brain data endpoints: /health, /brain/meta, /brain/points, row-index lookup. Route: delegated writer.
- [x] W2 Chat SSE: in-process MCP, agent streaming to events, highlight indices, conversation memory, /selections/{id}/indices, error paths, `flybrain-web`. Route: delegated writer.

## Acceptance criteria
- Points endpoint returns exactly count*12 bytes; decoding with numpy `<f4` matches the atlas.
- A fake model calling search_neurons produces tool_call -> highlight (correct indices) -> answer -> done, in order.
- Failures produce an `error` event and close the stream.
- `uv run pytest` green.

## Progress / evidence
- Engram mirror: pending (mem_save blocked by hook/auto-mode in this session).
- Versions: fastapi 0.162.3, uvicorn 0.54.0, httpx 0.28.1 (dev), starlette 1.10.0, mcp 1.30.0 (still <2), langgraph 1.2.20.
- Commits: ba89719 chore(deps); 4f49eb6 W1 binary points; cac8310 agent stream_events + checkpointer; 53a09de SSE chat; c59b958 flybrain-web.
- W1 RED: `tests/test_api_data.py` -> `ModuleNotFoundError: No module named 'flybrain.api'`; GREEN: 159 passed.
- agent RED: `ImportError: cannot import name 'stream_events' from 'flybrain.agent'`; GREEN: 166 passed.
- W2 RED: `tests/test_api_chat.py` -> 11 failed (`TypeError: create_app() got an unexpected keyword argument`); GREEN: 11 passed; full suite 182 passed after the `flybrain-web` tests (test_web.py was written alongside the code, not strictly red-first).
- Real server on the real TSV (no /chat): /health 139248 neurons, /brain/points 1670976 bytes = 139248*12, CORS header for localhost:5173, unknown selection 404.

## Decisions
- SSE: FastAPI 0.162 ships native SSE (`fastapi.sse.EventSourceResponse` + `ServerSentEvent`); used it, no sse-starlette.
- In-memory MCP: `mcp.shared.memory.create_connected_server_and_client_session(FastMCP)` entered in the lifespan via AsyncExitStack; tools via `load_mcp_tools(session)`. Works inside the lifespan, no fallback needed. The session is only opened when a model is configured.
- Checkpointer: `langgraph.checkpoint.memory.InMemorySaver` (MemorySaver is an alias); `build_graph(model, tools, checkpointer=None)`; `conversation_id` = `thread_id`.
- Tool-round budget is now per TURN (counted after the last human message), otherwise a long conversation would exhaust it.
- `agent.stream_events(graph, question, thread_id, max_steps)` yields tool_call / tool_result / answer / error dicts and never raises; the API only formats them and always appends `done`.
- Highlight threshold: 5000 indices inline (`max_inline_indices`), above that `indices: null` + `fetch`. One highlight per ticket per turn. Ids unknown to the atlas are ignored (RowIndex.lookup returns rows + missing count; `total` is the ticket size).
- tool_call events are emitted when the model asks; a call cut by the budget never runs and the turn ends with the hard-stop error.
- `flybrain-web` builds the model with init_chat_model before starting uvicorn; failure is one stderr line, exit 1.

- Live test with Gemini (2026-11-10): server streamed `meta` then only keep-alive pings for ~2 min while google-genai retried 503s (1, 2.9, 4.4, 8, 16.5 s backoff). Fix: whole-turn timeout in `stream_events(turn_timeout=...)`, API default 90 s via `FLYBRAIN_TURN_TIMEOUT`. RED: `TypeError: stream_events() got an unexpected keyword argument 'turn_timeout'` and `create_app() got an unexpected keyword argument 'turn_timeout'`; GREEN: 186 passed.

- Review (2026-11-10), 3 code slices after the deps commit, all granted, approved, acknowledged: A ba89719..cac8310 (367 lines) review-70a5e7a609d6a37e; B cac8310..53a09de (290) review-4414cf30f50f8d51; C 53a09de..d89630a (302) review-05e8954f997f108a.
- Follow-ups from review warnings (next PR): dangling tool_call left in the checkpointed thread when a turn is cut (hard stop/timeout) can poison the next turn (agent.py:205); TimeoutError raised by a tool with turn_timeout=None would format None (agent.py:206); FLYBRAIN_TURN_TIMEOUT not validated (api.py:94); README test count drift.

## Next step
Open PR, then T6 (Three.js viewer).
