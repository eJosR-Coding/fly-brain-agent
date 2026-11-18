"""Live session: the brain and the body advance together in chunks, closed loop, and every chunk is a message.

One chunk (default 50 ms of SIMULATED time): the brain advances (`BrainStream`), the spikes of the descending-neuron
groups become Hz, `DriveMapper` (OUR mapping, body_drive) turns them into the walking drive with its smoothing state
carried across chunks, the body advances (`BodyStream`) and a message is built. The session runs as fast as the compute
allows and REPORTS its speed in every message (`speed_factor` = simulated ms / wall ms of that chunk, `speed_label`
"x0.24 real time"); it never paces itself to pretend real time. Pure orchestration: the brain and body are injected
(fakes in the tests); `build_live_session` wires the real ones. The WebSocket route lives in api.py.

Thermal guard: before each chunk a GPU brain asks the `ThermalGuard`; above the limit the session moves to the CPU brain
for the rest of the session (the brain restarts from rest with the same stimulus; the body is untouched) and every
message says so (`fallback`). A GPU error does the same.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from flybrain import body_drive
from flybrain.body_readouts import dn_group_rows
from flybrain.simulation import ChunkSpikes

log = logging.getLogger(__name__)

CHUNK_MS = 50.0
MAX_ACTIVE_ROWS = 512  # per-chunk active rows sent when a client asks for them (strongest first)
SPEED_WINDOW = 10
BODY_MODES = ("kinematic", "physics")
NOTE = ("Live mode runs as fast as the computer allows and reports its real speed (speed_factor): it is NOT real time "
        "unless it says so. The brain is the connectome model, the DN -> walking-drive mapping is OUR choice "
        "(not published) and the body is NeuroMechFly, not the connectome.")


def speed_label(factor: float) -> str:
    return f"x{factor:.2f} real time"


def frames_to_bytes(frames: np.ndarray) -> bytes:
    """Frames (n, 48, 7) as little-endian int16 (positions in um and quaternions x 10000 both fit), clipped."""
    return np.clip(np.asarray(frames), -32768, 32767).astype("<i2").tobytes()


def frames_from_bytes(blob: bytes, shape) -> np.ndarray:
    return np.frombuffer(blob, dtype="<i2").reshape(shape).astype(np.int64)


class LiveSession:
    def __init__(self, brain, body, groups: dict[str, np.ndarray], *, chunk_ms: float = CHUNK_MS,
                 mapper: body_drive.DriveMapper | None = None, clock: Callable[[], float] = time.perf_counter,
                 thermal=None, brain_fallback: Callable[[], object] | None = None, include_active: bool = False,
                 max_active: int = MAX_ACTIVE_ROWS, binary_frames: bool = False,
                 brain_lock=None, body_lock=None) -> None:
        self.brain, self.body, self.groups = brain, body, groups
        self.chunk_ms, self.mapper = float(chunk_ms), mapper or body_drive.DriveMapper(chunk_ms)
        self._clock, self.thermal, self._brain_fallback = clock, thermal, brain_fallback
        self.include_active, self.max_active, self.binary_frames = include_active, max_active, binary_frames
        # callables returning a fresh context manager per chunk (SimulationService.exclusive, a lock's `lambda: lock`)
        self._brain_lock, self._body_lock = brain_lock or contextlib.nullcontext, body_lock or contextlib.nullcontext
        self.seq, self.t_ms = 0, 0.0
        self.fallback: str | None = None
        # "kinematic" = recorded physics gait blended live, "physics" = the MuJoCo body stepped live (see body_kinematic)
        self.body_mode = getattr(body, "mode", "physics")
        self.body_caption = getattr(body, "caption", "")
        self._stimulus: tuple[list[int] | None, float] = (None, 0.0)
        self._speeds: deque[float] = deque(maxlen=SPEED_WINDOW)
        self._closed = False
        # a cancelled websocket task leaves its worker thread finishing a chunk: close/step/stimulus never overlap
        self._lock = threading.RLock()

    # -- control -------------------------------------------------------------------------

    def set_stimulus(self, rows, rate_hz: float) -> None:
        """Poisson-drive atlas `rows` at `rate_hz` from the next chunk (None / 0 Hz clears). Invalid input raises
        SimulationError and leaves the session as it was."""
        arr = None if rows is None else [int(r) for r in rows]
        with self._lock, self._brain_lock():
            self.brain.set_stimulus(arr, float(rate_hz))
        self._stimulus = (arr, float(rate_hz))

    def close(self) -> None:
        with self._lock:
            self._close()

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for part in (self.brain, self.body):
            closer = getattr(part, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001 - closing must never raise out of a teardown
                    log.exception("closing a live part failed")

    def start_message(self) -> dict:
        return {"type": "started", "chunk_ms": self.chunk_ms, "backend": self.brain.backend, "fallback": self.fallback,
                "header": self.body.header, "body_mode": self.body_mode, "body_caption": self.body_caption,
                "dn_names": list(self.groups), "caption": body_drive.CAPTION,
                "note": NOTE, "binary_frames": self.binary_frames}

    # -- one chunk -----------------------------------------------------------------------

    def _to_cpu(self, why: str) -> None:
        old, self.brain = self.brain, self._brain_fallback()
        rows, rate = self._stimulus
        if rows:
            self.brain.set_stimulus(rows, rate)
        self.fallback = why
        log.warning("live session: brain moved to the CPU (%s)", why)
        old.close()

    def _advance_brain(self) -> ChunkSpikes:
        gpu = self.brain.backend == "gpu" and self._brain_fallback is not None
        if gpu and self.thermal is not None and not self.thermal.allow():
            t = self.thermal.temperature_c
            self._to_cpu(f"GPU too hot ({t:.0f} C, limit {self.thermal.limit_c:.0f} C): brain moved to the CPU for this session")
        with self._brain_lock():
            try:
                return self.brain.advance(self.chunk_ms)
            except Exception as exc:  # noqa: BLE001
                if self.brain.backend != "gpu" or self._brain_fallback is None:
                    raise
                self._to_cpu(f"GPU error ({exc}): brain moved to the CPU for this session")
            return self.brain.advance(self.chunk_ms)

    def step(self) -> dict:
        """Advance one chunk and return its message (frames as nested lists, or None when `binary_frames`)."""
        return self._step()[0]

    def step_binary(self) -> tuple[dict, bytes]:
        """Like `step`, with the frames as int16 bytes (the message says dtype and shape)."""
        msg, blob = self._step(force_binary=True)
        return msg, blob

    def _step(self, force_binary: bool = False) -> tuple[dict, bytes | None]:
        with self._lock:
            return self._step_locked(force_binary)

    def _step_locked(self, force_binary: bool) -> tuple[dict, bytes | None]:
        if self._closed:
            raise RuntimeError("the live session is closed")
        t0 = self._clock()
        chunk = self._advance_brain()
        t1 = self._clock()
        dn = {name: chunk.group_hz(rows) for name, rows in self.groups.items()}
        sample = self.mapper.step(dn)
        with self._body_lock():
            if getattr(self.body, "takes_command", False):  # kinematic body: the exact pace and turn, not a clipped drive
                out = self.body.advance(self.chunk_ms, sample.drive, sample.escape, pace=sample.pace, turn=sample.turn)
            else:
                out = self.body.advance(self.chunk_ms, sample.drive, sample.escape)
        t2 = self._clock()
        wall_ms = max((t2 - t0) * 1000.0, 1e-6)
        factor = self.chunk_ms / wall_ms
        self._speeds.append(factor)
        self.t_ms += self.chunk_ms
        stim_rows, stim_rate = self._stimulus
        frames = np.asarray(out["frames"])
        binary = force_binary or self.binary_frames
        msg = {
            "type": "chunk", "seq": self.seq, "t_ms": round(self.t_ms, 3), "chunk_ms": self.chunk_ms,
            "speed_factor": round(factor, 4), "speed_factor_avg": round(float(np.mean(self._speeds)), 4),
            "speed_label": speed_label(factor),
            "wall_ms": {"brain": round((t1 - t0) * 1000.0, 3), "body": round((t2 - t1) * 1000.0, 3),
                        "total": round(wall_ms, 3)},
            "dn": {k: round(float(v), 2) for k, v in dn.items()},
            "drive": [round(float(v), 4) for v in out["drive"][-1]] if len(out["drive"]) else [0.0, 0.0],
            "forward": round(sample.forward, 4), "turn": round(sample.turn, 4), "escape": bool(sample.escape),
            "active_count": int(len(chunk.active_rows)), "backend": self.brain.backend, "fallback": self.fallback,
            "stimulus": {"rows": len(stim_rows), "rate_hz": stim_rate} if stim_rows and stim_rate > 0 else None,
            "frames": None if binary else frames.tolist(),
            "thorax": {"pos_um": np.asarray(out["thorax"]["pos_um"]).tolist(),
                       "yaw_deg": np.asarray(out["thorax"]["yaw_deg"]).tolist()},
            "contact": np.asarray(out["contact"]).tolist(),
            "body_mode": self.body_mode, "body_caption": self.body_caption,
        }
        if out.get("world"):
            msg["world"] = out["world"]
        if binary:
            msg["frames_binary"] = {"dtype": "<i2", "shape": list(frames.shape)}
        if self.include_active:
            keep = np.argsort(-chunk.counts, kind="stable")[: self.max_active]
            msg["active"] = {"rows": [int(r) for r in chunk.active_rows[keep]],
                             "counts": [int(c) for c in chunk.counts[keep]], "total": int(len(chunk.active_rows))}
        self.seq += 1
        return msg, (frames_to_bytes(frames) if binary else None)


# -- wiring the real parts (api.py calls this in a worker thread) ----------------------------

def preset_rows(atlas, row_index, name: str) -> np.ndarray:
    """Atlas rows of a named preset (published neuron set); ValueError for an unknown name or an empty result."""
    from flybrain.tools import load_presets

    presets = load_presets()
    key = str(name).strip().lower()
    if key not in presets:
        raise ValueError(f"unknown preset {name!r}; available: {', '.join(sorted(presets))}")
    rows = row_index.positions(presets[key]["root_ids"])
    rows = rows[rows >= 0]
    if not len(rows):
        raise ValueError(f"none of the neurons of preset {key!r} exists in this atlas")
    return rows.astype(np.int64)


def readout_groups(atlas, row_index) -> dict[str, np.ndarray]:
    """The groups a live message reports: the DN rows the body follows (dng100, dna01/02 left/right, dnp01) + MN9."""
    groups = {k: v for k, v in dn_group_rows(atlas).items() if len(v)}
    try:
        groups["mn9"] = preset_rows(atlas, row_index, "mn9")
    except ValueError:
        pass
    return groups


def build_live_session(atlas, row_index, sims, body_lock=None, *, seed: int = 0, chunk_ms: float = CHUNK_MS,
                       include_active: bool = False, binary_frames: bool = False, body_mode: str = "kinematic") -> LiveSession:
    """The real brain (GPU when decided, usable and cool; else CPU) and the real body, ready to step.

    body_mode "kinematic" (default, fast: the recorded physics gait blended live, no flygym needed) or "physics"
    (MuJoCo stepped live, needs the `body` extra, slower). Raises BodyUnavailable (flygym or the motion library missing)
    or SimulationUnavailable (no weights) BEFORE anything heavy is built."""
    from flybrain import body_runner
    from flybrain.sim_service import SimulationUnavailable

    if body_mode not in BODY_MODES:
        raise ValueError(f"body_mode must be one of {BODY_MODES}")
    if body_mode == "physics":
        body_runner.require_flygym()
    if not sims.available:
        raise SimulationUnavailable("the brain simulator is not available (weights missing)")
    with sims.exclusive():
        brain = sims.open_stream(seed=seed)
    try:
        if body_mode == "kinematic":
            from flybrain.body_kinematic import KinematicBody, load_library

            try:
                body = KinematicBody(load_library())
            except FileNotFoundError as exc:
                raise body_runner.BodyUnavailable(f"the motion library is missing ({exc}); run scripts/build_motion_library.py") from exc
        else:
            body = body_runner.BodyStream(seed=seed)
    except Exception:
        brain.close()
        raise
    cpu = sims.simulation()
    return LiveSession(
        brain, body, readout_groups(atlas, row_index), chunk_ms=chunk_ms, thermal=sims.thermal,
        brain_fallback=lambda: cpu.open_stream(seed=seed), include_active=include_active,
        binary_frames=binary_frames, brain_lock=sims.exclusive, body_lock=(lambda: body_lock) if body_lock is not None else None,
    )
