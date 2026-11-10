# Feature: T6 Three.js brain viewer + chat

## Objective
A web page that shows all 139,248 neurons as a 3D point cloud and a chat panel; when the agent finds neurons, they light up live.

## Problem / why
The backend (T5) streams highlights as row indices, but nothing renders them. This is the visible, portfolio-grade part of the project.

## Design
- `web/`: Vite + TypeScript + three (pnpm). Dev server proxies `/api/*` -> `http://127.0.0.1:8000` (no CORS needed in dev).
- Brain: one `THREE.Points` with a `BufferGeometry`: `position` from `GET /brain/points` (`new Float32Array(buffer)`, contract from `/brain/meta`), `color` attribute (Float32Array rgb per point). Centered on the extent midpoint; FlyWire y grows downward, so flip y if the brain renders upside down (verify visually). OrbitControls.
- Highlight: on an SSE `highlight` event, recolor those indices and set `needsUpdate`; if `indices` is null, fetch `fetch` URL as `Uint32Array`. A "clear" action restores the base color.
- Chat: `fetch('/api/chat', {method: 'POST'})` + `ReadableStream` SSE parser (EventSource cannot POST). Render `tool_call`, `highlight` (with count), `answer`, `error`; keep `conversation_id` from `meta` for follow-up turns.
- No-LLM path (works when providers are down): backend `GET /neurons/search?cell_type=&nt=&super_class=&side=` returns {selection_id, total, indices|fetch} using the same tools + RowIndex; the UI has quick filter chips that call it.
- Look: black background, white/grey points, one accent color for highlights; minimal, matches the cover. No emojis.

## Scope
- In: backend search endpoint, `web/` app, unit tests for the SSE parser and binary decoding (vitest), README section, a real screenshot in the README.
- Out: production serving of the built app from FastAPI (later), auth, lesion/gain tools, mobile polish beyond basic responsiveness.

## Constraints
- Backend change via strict TDD (`uv run pytest`). Frontend: vitest unit tests for pure modules (SSE parser, decoders, color updates); visual check via a real browser screenshot.
- Dependencies: Python deps in their own `chore(deps)` commit; JS deps live in `web/package.json` + lockfile in their own `chore(deps)` commit.
- GitHub Flow; Claude merges after checks.

## TDD
- Mode: strict (source: global user config). Runners: `uv run pytest` (backend), `pnpm -C web test` (frontend units).

## Tasks
- [ ] V1 Backend `GET /neurons/search` (no LLM). Route: delegated writer.
- [ ] V2 Viewer: point cloud from the binary contract, orbit, highlight/clear, quick filters. Route: delegated writer.
- [ ] V3 Chat panel over SSE with live highlights; screenshot + README. Route: delegated writer.

## Acceptance criteria
- Page loads the real 139,248 points and renders them (screenshot proof).
- A quick filter (e.g. dopamine, left) lights up exactly the backend's indices.
- SSE parser handles events split across chunks; unit tests green.
- `uv run pytest` and `pnpm -C web test` green.

## Progress / evidence
- Engram mirror: pending (mem_save blocked by hook/auto-mode in this session).

## Next step
V1.
