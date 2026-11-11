"""HTTP adapter: brain data in binary, plus agent chat streamed over SSE.

The atlas loads ONCE, in the lifespan. The flybrain MCP server runs IN-PROCESS and the
agent reaches it over an in-memory MCP session, so the agent still only talks to tools
through MCP while the API shares the SAME SelectionStore (ticket -> root_ids -> rows).
"""

import asyncio
import json
import math
import os
import uuid
from contextlib import aclosing
from collections.abc import AsyncIterable
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import InMemorySaver
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import BaseModel, ConfigDict, StrictInt, StringConstraints

from flybrain.agent import MAX_STEPS, build_graph, keep_turns_from_env, stream_events
from flybrain.agent_mcp import SERVER_NAME
from flybrain.anatomy import NT_CODES, REGION_CODES, landmarks, nt_codes, region_codes
from flybrain.atlas import RowIndex, load_atlas
from flybrain.mcp_server import DEFAULT_ANNOTATIONS, build_server, load_connectome
from flybrain.experiments import dose_response
from flybrain.models import CHAT_MODEL_KWARGS, describe_models
from flybrain.selections import SelectionStore, UnknownSelectionError
from flybrain.silencing import silencing_screen
from flybrain.sim_service import SimulationService, SimulationUnavailable, UnknownSimulationError
from flybrain.tools import ACTIVE_HZ, load_presets, preset_selection, search_neurons, stimulate

BYTES_PER_POINT = 12  # 3 x float32
MAX_MESSAGE_CHARS = 2000
MAX_INLINE_INDICES = 5000  # bigger highlights are fetched as binary instead of inlined
TURN_TIMEOUT_S = 90.0  # provider SDKs can retry 503s for minutes; cap the whole chat turn
EXPERIMENT_TIMEOUT_S = 300.0  # extra turn time granted each time the model starts an experiment tool
EXPERIMENT_TOOLS = ("dose_response", "silencing_screen")
NEURON_PARTNERS = 5  # top partners per direction on the neuron card
MAX_ARG_CHARS = 100  # tool args in events are cut to keep events small
DEFAULT_CORS = "http://localhost:5173"  # Vite dev server


class ChatRequest(BaseModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_CHARS)]
    conversation_id: Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)] | None = None
    # must be one of the configured AND available ids (checked against the allowlist, see /models)
    model: Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)] | None = None


class SimulateRequest(BaseModel):
    """No-LLM simulation for the viewer chips: a preset or a ticket, plus optional readout/silence."""

    model_config = ConfigDict(extra="forbid")
    preset: str | None = None
    selection_id: str | None = None
    rate_hz: float = 100
    duration_ms: float = 500
    trials: StrictInt | None = None  # default: 3, or 5 when silencing
    readout_preset: str | None = None
    readout_selection_id: str | None = None
    silence_selection_id: str | None = None
    silence_preset: str | None = None


class _ExperimentRequest(BaseModel):
    """Shared by the no-LLM experiment endpoints: the stimulus and the readout, each a preset or a ticket."""

    model_config = ConfigDict(extra="forbid")
    preset: str | None = None
    selection_id: str | None = None
    readout_preset: str | None = None
    readout_selection_id: str | None = None
    duration_ms: float = 300
    trials: StrictInt = 3


class DoseResponseRequest(_ExperimentRequest):
    rates_hz: list[float] = [10, 25, 50, 100, 200]


class ScreenRequest(_ExperimentRequest):
    candidates: str = "top_non_sensory"
    n: StrictInt = 8
    rate_hz: float = 100


def _text_or_none(value) -> str | None:
    return value if isinstance(value, str) and value else None


def _short(value):
    if isinstance(value, str) and len(value) > MAX_ARG_CHARS:
        return value[:MAX_ARG_CHARS] + "…"
    if isinstance(value, dict):
        return {k: _short(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_short(v) for v in value[:20]]
    return value


def path_payload(data, row_index: RowIndex) -> dict | None:
    """A successful find_path result as {selection_id, rows, synapses, bottleneck}, or None.

    rows are atlas rows in route order, synapses has one count per hop (len(rows) - 1). A path with
    a neuron the atlas does not know, or without a single hop, is not drawable, so it gives None.
    """
    if not isinstance(data, dict) or data.get("found") is not True:
        return None
    path = data.get("path")
    if not isinstance(path, list) or len(path) < 2 or not all(isinstance(p, dict) for p in path):
        return None
    try:
        rows = row_index.positions([p["root_id"] for p in path])
        synapses = [int(p["synapses_from_previous"]) for p in path[1:]]
        bottleneck = int(data["min_hop_synapses"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (rows < 0).any():
        return None
    sid = data.get("selection_id")
    return {"selection_id": sid if isinstance(sid, str) else "", "rows": [int(r) for r in rows],
            "synapses": synapses, "bottleneck": bottleneck}


def _sse(event: str, data: dict) -> ServerSentEvent:
    return ServerSentEvent(event=event, data=data)


def _cors_origins() -> list[str]:
    raw = os.environ.get("FLYBRAIN_CORS_ORIGINS", DEFAULT_CORS)
    return [o.strip() for o in raw.split(",") if o.strip()]


def _resolve_turn_timeout(turn_timeout: float | None) -> float:
    """Argument, else $FLYBRAIN_TURN_TIMEOUT, else the default; must be a positive finite number."""
    if turn_timeout is not None:
        value, source = turn_timeout, "turn_timeout"
    elif os.environ.get("FLYBRAIN_TURN_TIMEOUT"):
        value, source = os.environ["FLYBRAIN_TURN_TIMEOUT"], "FLYBRAIN_TURN_TIMEOUT"
    else:
        return TURN_TIMEOUT_S
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = math.nan
    if not (math.isfinite(seconds) and seconds > 0):
        raise ValueError(f"{source} must be a positive number of seconds, got {value!r}")
    return seconds


def _resolve_experiment_timeout(experiment_timeout: float | None) -> float:
    """Argument, else $FLYBRAIN_EXPERIMENT_TIMEOUT, else the default; a positive finite number of seconds."""
    if experiment_timeout is not None:
        value, source = experiment_timeout, "experiment_timeout"
    elif os.environ.get("FLYBRAIN_EXPERIMENT_TIMEOUT"):
        value, source = os.environ["FLYBRAIN_EXPERIMENT_TIMEOUT"], "FLYBRAIN_EXPERIMENT_TIMEOUT"
    else:
        return EXPERIMENT_TIMEOUT_S
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = math.nan
    if not (math.isfinite(seconds) and seconds > 0):
        raise ValueError(f"{source} must be a positive number of seconds, got {value!r}")
    return seconds


async def _with_progress(events, sims):
    """Merge a turn's events with the progress of long experiments: yields ("event", ev) / ("progress", p).

    Progress arrives from the simulation thread; it is queued on the event loop next to the turn's
    own events, so the stream keeps its order. Closing this generator cancels the turn (which repairs
    its thread, see stream_events).
    """
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    off = sims.subscribe(lambda p: loop.call_soon_threadsafe(queue.put_nowait, ("progress", p)))

    async def pump() -> None:
        try:
            async for ev in events:
                await queue.put(("event", ev))
        finally:
            await queue.put(("end", None))

    task = asyncio.create_task(pump())
    try:
        while True:
            kind, item = await queue.get()
            if kind == "end":
                break
            yield kind, item
    finally:
        off()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def create_app(
    atlas_path=None,
    model=None,
    max_steps: int = MAX_STEPS,
    max_inline_indices: int = MAX_INLINE_INDICES,
    turn_timeout: float | None = None,
    model_name: str | None = None,
    connectome_path=None,
    simulation: SimulationService | None = None,
    models: list[str] | None = None,
    model_factory=None,
    env=None,
    keep_turns: int | None = None,
    experiment_timeout: float | None = None,
) -> FastAPI:
    """Build the app. ``model`` is a LangChain chat model; without one /chat reports an error.

    ``model_name`` is the provider:model label shown by /health (only while chat is on).
    ``models`` is the picker's registry of provider:model ids; ``model_factory(id)`` builds one
    on first use (default: init_chat_model) and ``env`` is where API keys are looked up.
    Only registry ids whose key is set can be chosen per message; every model shares ONE
    checkpointer, so a conversation keeps its history across a model switch."""
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
        app.state.regions = region_codes(atlas).tobytes()
        app.state.nt = nt_codes(atlas).tobytes()
        app.state.landmarks = landmarks(atlas)
        app.state.graph = None
        app.state.tools = None
        app.state.graphs = {}  # model id -> compiled graph, built lazily
        app.state.graph_lock = asyncio.Lock()
        app.state.connectome = load_connectome(atlas, connectome_path)  # optional: None if not built
        # the simulator loads lazily on the first run; only its presence is checked here
        app.state.sims = simulation if simulation is not None else SimulationService.default(atlas)
        async with AsyncExitStack() as stack:
            if model is not None:
                session = await stack.enter_async_context(
                    create_connected_server_and_client_session(
                        build_server(atlas, store, app.state.connectome, app.state.sims)
                    )
                )
                tools = await load_mcp_tools(session, server_name=SERVER_NAME)
                app.state.tools = tools
                app.state.checkpointer = InMemorySaver()
                app.state.graph = build_graph(model, tools, checkpointer=app.state.checkpointer,
                                                keep_turns=keep)
                app.state.graphs[default_id] = app.state.graph
            yield

    default_id = model_name or "default"
    registry = [default_id, *[m for m in (models or []) if m != default_id]]

    def _rows() -> list[dict]:
        """Registry rows; the default is built already, so it is available whatever the env says."""
        rows = describe_models(registry, os.environ if env is None else env)
        for row in rows:
            if row["id"] == default_id:
                row.pop("reason", None)
                row["available"] = True
        return rows

    def _make_model(model_id: str):
        if model_factory is not None:
            return model_factory(model_id)
        from langchain.chat_models import init_chat_model

        return init_chat_model(model_id, **CHAT_MODEL_KWARGS)

    async def _graph_for(model_id: str):
        """The graph for an allowlisted id, built once (all share the checkpointer)."""
        async with app.state.graph_lock:
            graph = app.state.graphs.get(model_id)
            if graph is None:
                chat_model = await asyncio.to_thread(_make_model, model_id)
                graph = build_graph(chat_model, app.state.tools, checkpointer=app.state.checkpointer,
                                       keep_turns=keep)
                app.state.graphs[model_id] = graph
            return graph

    app = FastAPI(title="flybrain", lifespan=lifespan)
    app.state.turn_timeout = _resolve_turn_timeout(turn_timeout)
    app.state.experiment_timeout = _resolve_experiment_timeout(experiment_timeout)
    app.state.keep_turns = keep = keep_turns_from_env() if keep_turns is None else keep_turns
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        chat_on = app.state.graph is not None
        return {
            "status": "ok",
            "neurons": len(app.state.atlas.neurons),
            "chat": chat_on,
            "model": model_name if chat_on else None,
            "connectome": app.state.connectome is not None,
            "connectome_edges": app.state.connectome.n_edges if app.state.connectome is not None else None,
            "simulation": app.state.sims.available,
        }

    @app.get("/models")
    def list_models():
        if app.state.graph is None:
            return {"default": None, "models": []}
        return {"default": default_id, "models": _rows()}

    async def chosen_model(req: ChatRequest) -> str | None:
        """Allowlist: an explicit model must be a configured, available id (else 422)."""
        if req.model is None:
            return None
        allowed = [r["id"] for r in _rows() if r["available"]]
        if app.state.graph is None or req.model not in allowed:
            raise HTTPException(
                status_code=422,
                detail={"message": f"model {req.model!r} is not available", "available": allowed},
            )
        return req.model

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
            "regions": {str(k): v for k, v in REGION_CODES.items()},
            "nt": {str(k): v for k, v in NT_CODES.items()},
        }

    @app.get("/brain/points")
    def brain_points():
        return Response(
            app.state.points,
            media_type="application/octet-stream",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @app.get("/brain/regions")
    def brain_regions():
        """One uint8 per atlas row (same order as /brain/points); the code table is in /brain/meta."""
        return Response(
            app.state.regions,
            media_type="application/octet-stream",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @app.get("/brain/nt")
    def brain_nt():
        """One uint8 per atlas row (same order as /brain/points): top neurotransmitter, table in /brain/meta."""
        return Response(
            app.state.nt,
            media_type="application/octet-stream",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @app.get("/brain/landmarks")
    def brain_landmarks():
        return app.state.landmarks

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

    def _record(simulation_id: str):
        try:
            return app.state.sims.get(simulation_id)
        except UnknownSimulationError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None

    @app.get("/simulations/{simulation_id}")
    def simulation_meta(simulation_id: str):
        rec = _record(simulation_id)
        # [] = nothing was silenced; None + silenced_truncated = too many to list inline
        listed = [] if rec.silenced_rows is None else rec.silenced_rows
        truncated = len(listed) > max_inline_indices
        silenced = None if truncated else listed.tolist() if hasattr(listed, "tolist") else listed
        return {"simulation_id": rec.id, "bin_ms": rec.bin_ms, "n_bins": rec.n_bins, "n_rows": rec.n_rows,
                "duration_ms": rec.duration_ms, "stimulus": rec.stimulus, "trials": rec.trials,
                "active_count": rec.active_count, "active_hz": ACTIVE_HZ, "compare_id": rec.compare_id,
                "silenced_rows": silenced, "silenced_truncated": truncated, "readout": rec.readout}

    @app.get("/simulations/{simulation_id}/rows")
    def simulation_rows(simulation_id: str, rows: str | None = None):
        """[{row, cell_type, super_class}] for the recorded rows (or only the comma-separated `rows` among them)."""
        rec = _record(simulation_id)
        wanted = [int(r) for r in rec.rows]
        if rows is not None:
            try:
                asked = {int(x) for x in rows.split(",") if x.strip()}
            except ValueError:
                raise HTTPException(status_code=422, detail="rows must be comma-separated integers") from None
            wanted = [r for r in wanted if r in asked]
        n = app.state.atlas.neurons
        cell_type, super_class = n["cell_type"].to_numpy(), n["super_class"].to_numpy()
        return [{"row": r, "cell_type": _text_or_none(cell_type[r]), "super_class": _text_or_none(super_class[r])}
                for r in wanted]

    @app.get("/simulations/{simulation_id}/frames")
    def simulation_frames(simulation_id: str):
        """uint32 LE atlas rows (n_rows), then uint8 spike counts (n_rows * n_bins, row-major)."""
        rec = _record(simulation_id)
        return Response(rec.rows.astype("<u4").tobytes() + rec.frames.tobytes(), media_type="application/octet-stream")

    def _ticket(preset: str | None, selection_id: str | None, what: str) -> str | None:
        if preset is not None and selection_id is not None:
            raise HTTPException(status_code=422, detail=f"give either {what}preset or {what}selection_id, not both")
        if preset is None:
            return selection_id
        res = preset_selection(app.state.atlas, app.state.store, preset)
        if res["selection_id"] is None:
            raise HTTPException(status_code=422, detail=res["hint"])
        return res["selection_id"]

    @app.post("/simulate")
    async def simulate(req: SimulateRequest):
        """Same summary as the stimulate tool, without an LLM. Runs in a thread (it takes seconds)."""
        if req.preset is None and req.selection_id is None:
            raise HTTPException(status_code=422, detail="give a preset or a selection_id")
        if not app.state.sims.available:
            raise HTTPException(status_code=503, detail=_unavailable_message())
        try:
            sid = _ticket(req.preset, req.selection_id, "")
            readout = _ticket(req.readout_preset, req.readout_selection_id, "readout_")
            silence = _ticket(req.silence_preset, req.silence_selection_id, "silence_")
            return await asyncio.to_thread(
                stimulate, app.state.atlas, app.state.store, app.state.sims, sid, req.rate_hz,
                req.duration_ms, req.trials, silence, readout,
            )
        except SimulationUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        except UnknownSelectionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    async def _run_experiment(fn, req: _ExperimentRequest, *args):
        if req.preset is None and req.selection_id is None:
            raise HTTPException(status_code=422, detail="give a preset or a selection_id")
        if req.readout_preset is None and req.readout_selection_id is None:
            raise HTTPException(status_code=422, detail="give a readout_preset or a readout_selection_id")
        if not app.state.sims.available:
            raise HTTPException(status_code=503, detail=_unavailable_message())
        try:
            sid = _ticket(req.preset, req.selection_id, "")
            readout = _ticket(req.readout_preset, req.readout_selection_id, "readout_")
            out = await asyncio.to_thread(fn, app.state.atlas, app.state.store, app.state.sims, sid, readout, *args)
            return {**out, "chart": app.state.sims.get_chart(out["chart_id"])}
        except SimulationUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        except UnknownSelectionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    @app.post("/experiments/dose_response")
    async def experiment_dose_response(req: DoseResponseRequest):
        """The dose_response tool without an LLM (viewer chip): the result plus its chart payload."""
        return await _run_experiment(dose_response, req, req.rates_hz, req.duration_ms, req.trials)

    @app.post("/experiments/screen")
    async def experiment_screen(req: ScreenRequest):
        """The silencing_screen tool without an LLM (viewer chip): the result plus its chart payload."""
        return await _run_experiment(
            silencing_screen, req, req.candidates, req.n, req.duration_ms, req.trials, req.rate_hz
        )

    @app.get("/experiments/progress")
    def experiment_progress():
        """{running, experiment, done, total} of the experiment in flight (the viewer polls it)."""
        snap = app.state.sims.progress_snapshot()
        return {"running": False} if snap is None else {"running": True, **snap}

    def _unavailable_message() -> str:
        try:
            app.state.sims.simulation()
        except SimulationUnavailable as exc:
            return str(exc)
        return "simulation unavailable"

    @app.get("/neurons/search")
    def neurons_search(cell_type: str | None = None, nt: str | None = None,
                       super_class: str | None = None, side: str | None = None):
        """No-LLM search for the viewer chips: same tool, same ticket store, same inline rule."""
        try:
            result = search_neurons(
                app.state.atlas, app.state.store, cell_type=cell_type, nt=nt, super_class=super_class, side=side
            )
        except ValueError as exc:  # filter validation; a TypeError here is a bug -> 500
            raise HTTPException(status_code=422, detail=str(exc)) from None
        body = {k: result[k] for k in ("selection_id", "total", "filters", "note", "hint") if k in result}
        sid = result["selection_id"]
        if sid is None:
            body["indices"] = []
        elif result["total"] > max_inline_indices:
            body |= {"indices": None, "fetch": f"/selections/{sid}/indices"}
        else:
            body["indices"] = _rows_for(sid)[0].tolist()
        return body

    @app.get("/neurons/{row}")
    def neuron_card(row: int):
        """One neuron by atlas row (the index the viewer uses): identity, position and top partners."""
        n = app.state.atlas.neurons
        if not 0 <= row < len(n):
            raise HTTPException(status_code=404, detail=f"no neuron at row {row}; rows run 0..{len(n) - 1}")
        rec = n.iloc[row]

        card = {
            "row": row,
            "root_id": str(int(rec["root_id"])),  # string: a root_id does not fit a JS number
            "cell_type": text_at(row, "cell_type"), "cell_class": text_at(row, "cell_class"),
            "super_class": text_at(row, "super_class"), "side": text_at(row, "side"),
            "top_nt": text_at(row, "top_nt"),
            "position_um": [float(rec["x_um"]), float(rec["y_um"]), float(rec["z_um"])],
            "region": REGION_CODES[int(np.frombuffer(app.state.regions, dtype=np.uint8)[row])],
        }
        conn = app.state.connectome
        card["partners"] = conn is not None
        if conn is not None:
            for key, direction in (("inputs", "in"), ("outputs", "out")):
                rows, syn = conn.neighbors([row], direction)
                card[key] = [
                    {"row": int(r), "cell_type": text_at(int(r), "cell_type"), "super_class": text_at(int(r), "super_class"),
                     "synapses": int(s)}
                    for r, s in zip(rows[:NEURON_PARTNERS], syn[:NEURON_PARTNERS])
                ]
                card[f"synapses_{direction}"] = int(syn.sum())
        return card

    def text_at(r: int, field: str):
        v = app.state.atlas.neurons[field].iat[r]
        return None if v is None or v != v else str(v)

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

    def _activity(content: str) -> ServerSentEvent | None:
        """A tool result carrying a simulation_id also becomes an activity event (the animation)."""
        try:
            data = json.loads(content)
            sim_id = data.get("simulation_id") if isinstance(data, dict) else None
            rec = app.state.sims.get(sim_id) if isinstance(sim_id, str) else None
        except (ValueError, UnknownSimulationError):
            return None
        if rec is None:
            return None
        return _sse("activity", {"simulation_id": rec.id, "n_bins": rec.n_bins, "bin_ms": rec.bin_ms,
                                "active_count": rec.active_count, "active_hz": ACTIVE_HZ,
                                "compare_id": rec.compare_id})

    def _chart(content: str) -> ServerSentEvent | None:
        """An experiment result carrying a chart_id also becomes a chart event (the plot)."""
        try:
            data = json.loads(content)
            chart_id = data.get("chart_id") if isinstance(data, dict) else None
            chart = app.state.sims.get_chart(chart_id) if isinstance(chart_id, str) else None
        except (ValueError, UnknownSimulationError):
            return None
        return None if chart is None else _sse("chart", chart)

    def _path(tool: str, content: str) -> ServerSentEvent | None:
        """find_path also becomes a path event: the route as atlas rows, for the cables."""
        if tool != "find_path":
            return None
        try:
            payload = path_payload(json.loads(content), app.state.row_index)
        except ValueError:
            return None
        return None if payload is None else _sse("path", payload)

    @app.post("/chat", response_class=EventSourceResponse)
    async def chat(
        req: ChatRequest, picked: Annotated[str | None, Depends(chosen_model)]
    ) -> AsyncIterable[ServerSentEvent]:
        conversation_id = req.conversation_id or uuid.uuid4().hex
        used = picked or default_id
        yield _sse("meta", {"conversation_id": conversation_id, "model": used if app.state.graph else None})
        graph = app.state.graph
        if graph is not None and picked is not None:
            try:
                graph = await _graph_for(picked)
            except Exception as exc:  # missing provider package, bad key format...
                reason = " ".join(str(exc).split()) or type(exc).__name__
                yield _sse("error", {"message": f"could not set up model {picked}: {reason}"})
                graph = False
        if graph is None:
            yield _sse("error", {"message": "no chat model configured on this server"})
        elif graph is not False:
            seen: set[str] = set()
            # stream_events never raises: failures arrive as an "error" event
            # aclosing: when the client leaves, the stream is closed right away (not at GC time)
            # so its finally block repairs the thread before the next turn.
            extensions = {name: app.state.experiment_timeout for name in EXPERIMENT_TOOLS}
            running_experiment = False  # between an experiment tool's call and its result: progress belongs to this turn
            async with aclosing(_with_progress(
                stream_events(graph, req.message, conversation_id, max_steps, app.state.turn_timeout, used, extensions),
                app.state.sims,
            )) as merged:
                async for kind, ev in merged:
                    if kind == "progress":
                        if running_experiment:
                            yield _sse("progress", ev)
                        continue
                    if ev["type"] == "tool_call":
                        running_experiment = ev["tool"] in EXPERIMENT_TOOLS
                        yield _sse("tool_call", {"id": ev["id"], "tool": ev["tool"], "args": _short(ev["args"])})
                    elif ev["type"] == "answer_delta":
                        yield _sse("answer_delta", {"text": ev["text"]})
                    elif ev["type"] == "tool_result":
                        running_experiment = False
                        yield _sse("tool_result", {"id": ev["id"], "tool": ev["tool"], "ok": not ev["error"], "ms": ev["ms"]})
                        hl = None if ev["error"] else _highlight(ev["content"], seen)
                        if hl is not None:
                            yield hl
                        route = None if ev["error"] else _path(ev["tool"], ev["content"])
                        if route is not None:
                            yield route
                        act = None if ev["error"] else _activity(ev["content"])
                        if act is not None:
                            yield act
                        chart = None if ev["error"] else _chart(ev["content"])
                        if chart is not None:
                            yield chart
                    elif ev["type"] == "usage":
                        yield _sse("usage", {k: ev[k] for k in
                                             ("calls", "input_tokens", "output_tokens", "total_tokens")})
                    elif ev["type"] == "answer":
                        yield _sse("answer", {"text": ev["text"]})
                    else:
                        yield _sse("error", {"message": ev["message"]})
        yield _sse("done", {})  # always last, so the client never waits on a dead stream

    return app
