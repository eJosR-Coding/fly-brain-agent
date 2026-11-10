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
- [x] V1 Backend `GET /neurons/search` (no LLM). Route: delegated writer.
- [x] V2 Viewer: point cloud from the binary contract, orbit, highlight/clear, quick filters. Route: delegated writer.
- [x] V3 Chat panel over SSE with live highlights; screenshot + README. Route: delegated writer.

## Acceptance criteria
- Page loads the real 139,248 points and renders them (screenshot proof).
- A quick filter (e.g. dopamine, left) lights up exactly the backend's indices.
- SSE parser handles events split across chunks; unit tests green.
- `uv run pytest` and `pnpm -C web test` green.

## Progress / evidence
- V1 (cbc404f): RED `tests/test_api_search.py` 5 failed (404, route missing) -> GREEN 212 passed (`uv run pytest -q`). Zero matches: 200, selection_id null, indices [], hint; bad filter: 422.
- V2 (9bec0a3 deps, c760664): vitest RED = 3 files failed (modules missing: sse, colors, brainData) -> GREEN 19 passed. Versions: three 0.187.0, vite 8.3.7, vitest 5.1.0. Center computed from min/max of the points (meta.extent_um is a size, not bounds). y flipped via a wrapper group scale.y=-1; confirmed: dopamine PAM cluster shows at the top.
- V3 (22faaa7): RED chat.test.ts "Cannot find module ../src/chat" -> GREEN 23 passed across 4 files. SSE malformed JSON policy: emitted as an `error` event, never throws.
- Visual: real backend without a model + `vite preview`, headless chromium (swiftshader). Screenshot assets/viewer.png (317 KB): 139,248 points loaded, 2,926 left dopamine neurons lit white (chat stream mocked in the browser, highlight indices real), input re-enabled after done.
- README updated (Viewer section, roadmap, badge 235 = 212 python + 23 web).
- Engram mirror: pending (mem_save blocked by hook/auto-mode in this session).

- Review (2026-11-10), 4 slices, all granted, approved, acknowledged: P 3133884..49ec384 (PR #4 turn-cleanup fixes, 314 lines) review-70175b8a2011f4f7; Q f988eb3..0947065 (search endpoint, 119) review-b04a409fc9cc9f2b; R 0947065..73c2d9e (viewer core, 630) review-706814be4caa0000; S 73c2d9e..HEAD (chat + docs, 234) review-1170209bdb70821a. Deps/scaffold commit f988eb3 moved first (user-authorized rebase).
- Follow-ups from review warnings (next fix PR): client disconnect (CancelledError) skips the dangling-call repair (agent.py:273); repair runs before the error event so a slow repair delays it (agent.py:273); broad TypeError -> 422 in /neurons/search (api.py:167); viewer: chip clicks can race and show a stale selection (main.ts:46-73), unvalidated highlight payload can abort the chat stream (chat.ts:51), meta without conversation_id (chat.ts:49), SSE parser buffer edge (sse.ts:18), brainData byte checks (brainData.ts:54).

## Next step
Parent: review, then PR. Penpot-driven visual restyle only touches web/src/style.css (theme.ts reads its custom properties).
