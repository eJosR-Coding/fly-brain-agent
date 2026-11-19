"""The /ws/live WebSocket: one live brain+body session at a time (see flybrain.live for the session itself).

Protocol (JSON text frames; close codes are 4xxx application codes, the reason says why):
  client -> {"type":"start","preset":name|null,"rate_hz":n,["seed":int,"chunk_ms":50,"include_active":bool,"binary_frames":bool,"body_mode":"kinematic"|"physics","vision":"arena"|null,"background_hz":n]}
            {"type":"stimulus","preset":name|null | "rows":[atlas rows],"rate_hz":n}   (null/no rows clears the drive)
            {"type":"ping"} (keeps an idle session alive)   {"type":"stop"}
  server -> {"type":"started",...,"arena":{"kind":"glass_cube","interior_mm","bounds_mm":[xmin,xmax,ymin,ymax],"note"}}
            (the server's glass cube, fly frame at the start: x forward, y left, mm; every chunk's "world" reports
            blocked / walls);   {"type":"chunk",...} (every chunk; with binary_frames a binary frame of int16 frames
            follows each chunk message)  {"type":"stimulus","ok":true,...}  {"type":"error","detail":...}
            {"type":"stopped","reason":"client"|"idle"}
Close codes: 4400 bad request, 4403 origin not allowed, 4408 idle timeout, 4409 another live session is running,
4503 the body (flygym) or the brain weights are unavailable, 1011 the session failed.
A new start right after a client stop/disconnect waits (at most BUSY_WAIT_S) for the previous teardown; a session that is
really running is refused at once with 4409.
The heavy work runs in a worker thread (`asyncio.to_thread`), so the event loop stays free between chunks. A client that
disconnects ends the session after at most one chunk; a session with no client message for `idle_s` is stopped.
"""

from __future__ import annotations

import asyncio
import json
import threading
import logging
import math
import time
from collections.abc import Callable

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect

from flybrain import body_runner
from flybrain.fly_eye import VisionUnavailable
from flybrain.live import CHUNK_MS, MAX_BACKGROUND_HZ, build_live_session, preset_rows
from flybrain.sim_service import SimulationUnavailable
from flybrain.simulation import SimulationError

log = logging.getLogger(__name__)

IDLE_ENV = "FLYBRAIN_LIVE_IDLE_MIN"
DEFAULT_IDLE_S = 300.0
BUSY_WAIT_S = 3.0  # a new start waits at most this long for the previous session's teardown
MAX_RATE_HZ = 10_000.0
MAX_STIM_ROWS = 200_000
LEAD_S = 0.25  # a session that outruns real time waits: the viewer buffers ~6 chunks and would drop the surplus
CODE_BAD_REQUEST, CODE_ORIGIN, CODE_IDLE, CODE_BUSY, CODE_UNAVAILABLE = 4400, 4403, 4408, 4409, 4503


class BadRequest(ValueError):
    """A malformed client message (the text goes back to the client)."""


def resolve_idle_s(idle_s: float | None, environ) -> float:
    if idle_s is not None:
        value, source = idle_s, "live_idle_s"
    elif environ.get(IDLE_ENV):
        try:
            value, source = float(environ[IDLE_ENV]) * 60.0, IDLE_ENV
        except ValueError:
            raise ValueError(f"{IDLE_ENV} must be a number of minutes, got {environ[IDLE_ENV]!r}") from None
    else:
        return DEFAULT_IDLE_S
    if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
        raise ValueError(f"{source} must be a positive number of seconds, got {value!r}")
    return float(value)


def _number(msg: dict, key: str, default: float) -> float:
    value = msg.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= MAX_RATE_HZ:
        raise BadRequest(f"{key} must be a number within 0..{MAX_RATE_HZ:g}, got {value!r}")
    return float(value)


def _stimulus_rows(msg: dict, atlas, row_index):
    """(rows or None, rate) of a start/stimulus message: explicit atlas rows, a preset, or nothing (clear)."""
    rate = _number(msg, "rate_hz", 100.0 if (msg.get("preset") or msg.get("rows")) else 0.0)
    rows = msg.get("rows")
    if rows is not None:
        if (not isinstance(rows, list) or len(rows) > MAX_STIM_ROWS
                or any(isinstance(r, bool) or not isinstance(r, int) or r < 0 for r in rows)):
            raise BadRequest(f"rows must be a list of at most {MAX_STIM_ROWS} atlas rows (non-negative integers)")
        return (rows or None), rate
    preset = msg.get("preset")
    if preset is None:
        return None, rate
    if not isinstance(preset, str):
        raise BadRequest("preset must be a name or null")
    try:
        return [int(r) for r in preset_rows(atlas, row_index, preset)], rate
    except ValueError as exc:
        raise BadRequest(str(exc)) from None


def _start_options(msg: dict) -> dict:
    seed = msg.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BadRequest("seed must be an integer")
    chunk_ms = msg.get("chunk_ms", CHUNK_MS)
    try:
        body_runner.chunk_frames(chunk_ms)
    except (ValueError, TypeError):
        raise BadRequest("chunk_ms must be a whole number of 5 ms frames") from None
    if not 5 <= chunk_ms <= 500 or round(chunk_ms * 10) % 1:  # also a multiple of the brain's 0.1 ms tick
        raise BadRequest("chunk_ms must be within 5..500")
    for key in ("include_active", "binary_frames"):
        if not isinstance(msg.get(key, False), bool):
            raise BadRequest(f"{key} must be true or false")
    body_mode = msg.get("body_mode", "kinematic")
    if body_mode not in ("kinematic", "physics"):
        raise BadRequest('body_mode must be "kinematic" or "physics"')
    vision = msg.get("vision")
    if vision not in (None, "arena"):
        raise BadRequest('vision must be "arena" or null')
    background = msg.get("background_hz", 0)
    if isinstance(background, bool) or not isinstance(background, (int, float)) or not 0 <= background <= MAX_BACKGROUND_HZ:
        raise BadRequest(f"background_hz must be a number within 0..{MAX_BACKGROUND_HZ:g}")
    return {"seed": seed, "chunk_ms": float(chunk_ms), "include_active": msg.get("include_active", False),
            "binary_frames": msg.get("binary_frames", False), "body_mode": body_mode, "vision": vision,
            "background_hz": float(background)}


def register_live(app: FastAPI, origins: Callable[[], list[str]], idle_s: float, factory=None) -> None:
    """Add /ws/live to `app`. `factory(atlas, row_index, sims, body_lock, **options)` builds the session (injectable)."""
    app.state.live_lock = threading.Lock()
    app.state.live_idle_s = idle_s
    app.state.live_closing = False  # True from the moment the running session starts tearing down
    app.state.live_busy_wait_s = BUSY_WAIT_S
    app.state.live_stream_note = None
    app.state.live_session = None  # the running session, for the chat's world tools
    build = factory or build_live_session

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket) -> None:
        await ws.accept()
        origin = ws.headers.get("origin")
        if origin is not None and origin not in origins():
            await ws.close(code=CODE_ORIGIN, reason="origin not allowed")
            return
        # one session at a time. A session that is only TEARING DOWN (client stop/disconnect: the GPU state is freed in a
        # worker thread) is waited for, bounded; a session that is really running is refused at once.
        got = app.state.live_lock.acquire(blocking=False)
        if not got and app.state.live_closing:
            got = await asyncio.to_thread(app.state.live_lock.acquire, True, app.state.live_busy_wait_s)
        if not got:
            await ws.close(code=CODE_BUSY, reason="a live session is already running")
            return
        app.state.live_closing = False
        session = None
        try:
            session = await _open(ws, app, build, idle_s)
            if session is None:
                return
            app.state.live_session = session
            await _stream(ws, app, session, idle_s)
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            log.exception("live session failed")
            await _safe(ws.send_json({"type": "error", "detail": f"{type(exc).__name__}: {exc}"}))
            await _safe(ws.close(code=1011, reason="live session failed"))
        finally:
            app.state.live_session = None
            app.state.live_closing = True  # before the first await: a restart arriving now waits for the lock

            def teardown() -> None:
                # in a worker thread, so a cancelled task (client gone) cannot skip it: frees the GPU state, then the lock
                try:
                    if session is not None:
                        session.close()
                finally:
                    app.state.live_lock.release()

            await asyncio.shield(asyncio.get_running_loop().run_in_executor(None, teardown))


async def _safe(coro) -> None:
    try:
        await coro
    except Exception:  # noqa: BLE001 - the peer is already gone
        pass


async def _fail(ws: WebSocket, code: int, detail: str) -> None:
    await _safe(ws.send_json({"type": "error", "detail": detail}))
    await _safe(ws.close(code=code, reason=detail.encode()[:120].decode(errors="ignore")))


async def _open(ws: WebSocket, app: FastAPI, build, idle_s: float):
    """Wait for the start message and build the session in a worker thread; None when the socket was closed."""
    try:
        first = json.loads(await asyncio.wait_for(ws.receive_text(), idle_s))
    except asyncio.TimeoutError:
        await _fail(ws, CODE_IDLE, "no start message")
        return None
    except json.JSONDecodeError:
        await _fail(ws, CODE_BAD_REQUEST, "the first message must be JSON")
        return None
    try:
        if not isinstance(first, dict) or first.get("type") != "start":
            raise BadRequest('the first message must be {"type": "start", ...}')
        rows, rate = _stimulus_rows(first, app.state.atlas, app.state.row_index)
        options = _start_options(first)
    except BadRequest as exc:
        await _fail(ws, CODE_BAD_REQUEST, str(exc))
        return None
    try:
        session = await asyncio.to_thread(build, app.state.atlas, app.state.row_index, app.state.sims,
                                          app.state.body_lock, **options)
    except (body_runner.BodyUnavailable, SimulationUnavailable, VisionUnavailable) as exc:
        await _fail(ws, CODE_UNAVAILABLE, str(exc))
        return None
    try:
        if rows:
            session.set_stimulus(rows, rate)
    except SimulationError as exc:
        await asyncio.to_thread(session.close)
        await _fail(ws, CODE_BAD_REQUEST, str(exc))
        return None
    app.state.live_stream_note = getattr(app.state.sims, "stream_note", None)
    await ws.send_json(session.start_message())
    return session


async def _stream(ws: WebSocket, app: FastAPI, session, idle_s: float) -> None:
    commands: asyncio.Queue = asyncio.Queue()
    gone = asyncio.Event()
    last_seen = time.monotonic()

    async def read() -> None:
        nonlocal last_seen
        try:
            while True:
                text = await ws.receive_text()
                last_seen = time.monotonic()
                try:
                    cmd = json.loads(text)
                except json.JSONDecodeError:
                    cmd = None
                if isinstance(cmd, dict) and cmd.get("type") == "stop":
                    # mark the session closing NOW, not after the chunk in flight: a client that sends stop and
                    # reconnects at once (the browser does) must wait for the lock instead of getting 4409
                    app.state.live_closing = True
                await commands.put(cmd)
        except (WebSocketDisconnect, RuntimeError):
            app.state.live_closing = True  # the client is gone: same reason as a stop
        finally:
            gone.set()

    reader = asyncio.create_task(read())
    t_start = time.monotonic()
    try:
        while not gone.is_set():
            while not commands.empty():
                cmd = commands.get_nowait()
                if isinstance(cmd, dict) and cmd.get("type") == "stop":
                    await ws.send_json({"type": "stopped", "reason": "client"})
                    await ws.close(code=1000, reason="stopped")
                    return
                await _handle(ws, app, session, cmd)
            if time.monotonic() - last_seen > idle_s:
                await ws.send_json({"type": "stopped", "reason": "idle"})
                await ws.close(code=CODE_IDLE, reason="idle timeout")
                return
            if session.binary_frames:
                msg, blob = await asyncio.to_thread(session.step_binary)
            else:
                msg, blob = await asyncio.to_thread(session.step), None
            msg["live"] = {"idle_s_left": round(max(0.0, idle_s - (time.monotonic() - last_seen)), 1)}
            await ws.send_json(msg)
            if blob is not None:
                await ws.send_bytes(blob)
            sim_s = getattr(session, "t_ms", None)
            if isinstance(sim_s, (int, float)):  # faster than real time: stay at most LEAD_S ahead (slower is never padded)
                ahead = sim_s / 1000.0 - (time.monotonic() - t_start) - LEAD_S
                if ahead > 0:
                    await asyncio.sleep(ahead)
            await asyncio.sleep(0)  # let the reader task run between chunks
    finally:
        reader.cancel()


async def _handle(ws: WebSocket, app: FastAPI, session, cmd) -> None:
    """One non-stop client command; problems become {"type": "error"} messages and never end the session."""
    try:
        kind = cmd.get("type") if isinstance(cmd, dict) else None
        if kind == "ping":
            return
        if kind == "stimulus":
            rows, rate = _stimulus_rows(cmd, app.state.atlas, app.state.row_index)
            await asyncio.to_thread(session.set_stimulus, rows, rate)
            await ws.send_json({"type": "stimulus", "ok": True, "rows": len(rows or []),
                                "rate_hz": rate if rows else 0.0})
            return
        raise BadRequest(f"unknown or malformed command {kind!r}; use start, stimulus, ping or stop")
    except (BadRequest, SimulationError) as exc:
        await ws.send_json({"type": "error", "detail": str(exc), "command": cmd.get("type") if isinstance(cmd, dict) else None})
