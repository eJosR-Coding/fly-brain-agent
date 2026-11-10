"""HTTP adapter: brain data in binary, plus agent chat streamed over SSE.

The atlas loads ONCE, in the lifespan. The flybrain MCP server runs IN-PROCESS and the
agent reaches it over an in-memory MCP session, so the agent still only talks to tools
through MCP while the API shares the SAME SelectionStore (ticket -> root_ids -> rows).
"""

import json
import os
import uuid
from collections.abc import AsyncIterable
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import InMemorySaver
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import BaseModel, StringConstraints

from flybrain.agent import MAX_STEPS, build_graph, stream_events
from flybrain.agent_mcp import SERVER_NAME
from flybrain.atlas import RowIndex, load_atlas
from flybrain.mcp_server import DEFAULT_ANNOTATIONS, build_server
from flybrain.selections import SelectionStore, UnknownSelectionError

BYTES_PER_POINT = 12  # 3 x float32
MAX_MESSAGE_CHARS = 2000
MAX_INLINE_INDICES = 5000  # bigger highlights are fetched as binary instead of inlined
MAX_ARG_CHARS = 100  # tool args in events are cut to keep events small
DEFAULT_CORS = "http://localhost:5173"  # Vite dev server


class ChatRequest(BaseModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_CHARS)]
    conversation_id: Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)] | None = None


def _short(value):
    if isinstance(value, str) and len(value) > MAX_ARG_CHARS:
        return value[:MAX_ARG_CHARS] + "…"
    if isinstance(value, dict):
        return {k: _short(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_short(v) for v in value[:20]]
    return value


def _sse(event: str, data: dict) -> ServerSentEvent:
    return ServerSentEvent(event=event, data=data)


def _cors_origins() -> list[str]:
    raw = os.environ.get("FLYBRAIN_CORS_ORIGINS", DEFAULT_CORS)
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app(
    atlas_path=None,
    model=None,
    max_steps: int = MAX_STEPS,
    max_inline_indices: int = MAX_INLINE_INDICES,
) -> FastAPI:
    """Build the app. ``model`` is a LangChain chat model; without one /chat reports an error."""
    path = Path(atlas_path or os.environ.get("FLYBRAIN_ANNOTATIONS") or DEFAULT_ANNOTATIONS)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        atlas = load_atlas(path)
        store = SelectionStore()
        app.state.atlas = atlas
        app.state.store = store
        app.state.row_index = RowIndex(atlas)
        # little-endian float32, xyz interleaved, atlas row order: GPU-ready, built once
        xyz = atlas.neurons[["x_um", "y_um", "z_um"]].to_numpy()
        app.state.points = np.ascontiguousarray(xyz.astype("<f4")).tobytes()
        app.state.graph = None
        async with AsyncExitStack() as stack:
            if model is not None:
                session = await stack.enter_async_context(
                    create_connected_server_and_client_session(build_server(atlas, store))
                )
                tools = await load_mcp_tools(session, server_name=SERVER_NAME)
                app.state.graph = build_graph(model, tools, checkpointer=InMemorySaver())
            yield

    app = FastAPI(title="flybrain", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "neurons": len(app.state.atlas.neurons)}

    @app.get("/brain/meta")
    def brain_meta():
        count = len(app.state.atlas.neurons)
        return {
            "count": count,
            "dtype": "float32",
            "endianness": "little",
            "layout": "xyz-interleaved",
            "order": "atlas rows",
            "extent_um": list(app.state.atlas.extent_um()),
            "bytes": count * BYTES_PER_POINT,
        }

    @app.get("/brain/points")
    def brain_points():
        return Response(
            app.state.points,
            media_type="application/octet-stream",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    def _rows_for(selection_id: str) -> tuple[np.ndarray, int]:
        selection = app.state.store.get(selection_id)
        rows, _missing = app.state.row_index.lookup(selection.root_ids)
        return rows, len(selection.root_ids)

    @app.get("/selections/{selection_id}/indices")
    def selection_indices(selection_id: str):
        try:
            rows, _ = _rows_for(selection_id)
        except UnknownSelectionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        return Response(rows.tobytes(), media_type="application/octet-stream")

    def _highlight(content: str, seen: set[str]) -> ServerSentEvent | None:
        """A tool result carrying a selection_id becomes a highlight (once per ticket)."""
        try:
            data = json.loads(content)
        except ValueError:
            return None
        sid = data.get("selection_id") if isinstance(data, dict) else None
        if not isinstance(sid, str) or sid in seen:
            return None
        try:
            rows, total = _rows_for(sid)
        except UnknownSelectionError:
            return None
        seen.add(sid)
        payload = {"selection_id": sid, "total": total}
        if total > max_inline_indices:
            payload |= {"indices": None, "fetch": f"/selections/{sid}/indices"}
        else:
            payload["indices"] = rows.tolist()
        return _sse("highlight", payload)

    @app.post("/chat", response_class=EventSourceResponse)
    async def chat(req: ChatRequest) -> AsyncIterable[ServerSentEvent]:
        conversation_id = req.conversation_id or uuid.uuid4().hex
        yield _sse("meta", {"conversation_id": conversation_id})
        graph = app.state.graph
        if graph is None:
            yield _sse("error", {"message": "no chat model configured on this server"})
        else:
            seen: set[str] = set()
            # stream_events never raises: failures arrive as an "error" event
            async for ev in stream_events(graph, req.message, conversation_id, max_steps):
                if ev["type"] == "tool_call":
                    yield _sse("tool_call", {"tool": ev["tool"], "args": _short(ev["args"])})
                elif ev["type"] == "tool_result":
                    hl = None if ev["error"] else _highlight(ev["content"], seen)
                    if hl is not None:
                        yield hl
                elif ev["type"] == "answer":
                    yield _sse("answer", {"text": ev["text"]})
                else:
                    yield _sse("error", {"message": ev["message"]})
        yield _sse("done", {})  # always last, so the client never waits on a dead stream

    return app
