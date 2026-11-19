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
import json
import logging
import threading
from pathlib import Path
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from flybrain import body_drive
from flybrain import fly_eye
from flybrain import live_world
from flybrain.fly_eye import VISION_LABEL
from flybrain.body_readouts import dn_group_rows
from flybrain.simulation import ChunkSpikes

log = logging.getLogger(__name__)

CHUNK_MS = 50.0
MAX_ACTIVE_ROWS = 512  # per-chunk active rows sent when a client asks for them (strongest first)
SPEED_WINDOW = 10
BODY_MODES = ("kinematic", "physics")
# The arena: ONE source of truth, the server's. A glass cube centred on the fly's start (the origin of the reported root),
# standing on the floor plane. Axes of the bounds: the fly's frame at the start, x forward, y left, in mm; the viewer maps
# x -> its Z and y -> its X. Kinematic mode walks the fly on any glass surface (body_surface.py, a world rule); physics mode clamps the REPORTED root.
ARENA_INTERIOR_MM = 50.0
ARENA_NOTE = "world rule: the fly walks on any glass surface (adhesive pads); not a brain decision"
PHYSICS_ARENA_NOTE = ("World rule (not the brain): the glass wall clamps the displayed root (display clamp: the physics body "
                      "itself is not stopped, its legs keep stepping).")
VISION_CAPTION = (
    "Vision is OURS: the fly's motion through the glass cube becomes T4/T5 firing rates (Reichardt correlators, photoreceptor "
    "path bypassed); what the fly does is whatever the brain does with that input, nothing is scripted."
)
TOUCH_HZ = 25.0  # OUR constant rate for antennal mechanosensors when the body is blocked by a wall (labelled, not validated). 100 Hz fired DNp01 ~12 Hz and froze the fly in the escape pose at the wall (3/3 seeds); 25 Hz does not (scratch comparison, 3 seeds)
TOUCH_HEAD_ON_DEG = 30.0  # a wall within this angle of the heading touches both antennae
TOUCH_MS = 100.0  # OUR duration of the touch at a surface transition (the brief contact when the fly meets the next glass surface; labelled, not validated)
TOUCH_NOTE = (f"Touch is OURS (not validated): for {TOUCH_MS:g} ms at a surface transition (the fly meets the next glass surface), the antennal mechanosensors (Johnston's organ JO-* "
              f"and antennal bristles BM_Ant) on the side of the contact fire at a constant {TOUCH_HZ:g} Hz (both when head-on); the leg "
              "touch route is in leg_note; what the fly does is the brain's")
LEG_TOUCH_HZ = TOUCH_HZ  # OUR constant rate for the matched ascending neurons of the legs on the contact side (labelled, not validated)
LEG_TOUCH_JSON = Path(__file__).parent / "data" / "leg_touch_ans.json"
LEG_TOUCH_NOTE = (f"Leg touch is OURS (not validated): the leg mechanosensory route is leg bristle / hair-plate neurons -> nerve cord -> ascending "
                  f"neurons (BANC v888 connectivity, <= 2 synapses) -> brain; the brain-only FlyWire connectome starts at the ascending neurons, so "
                  f"the matched ascending neurons of the legs on the contact side (front legs only when head-on) fire at a constant {LEG_TOUCH_HZ:g} Hz")
_WALL_NORMAL_DEG = {"x_max": 0.0, "y_max": 90.0, "x_min": 180.0, "y_min": -90.0}  # outward normal of each wall, arena frame


def load_leg_touch(path: Path = LEG_TOUCH_JSON) -> dict[str, dict[str, list[int]]]:
    """{'left'|'right': {'front': root ids, 'all': root ids}} from the packaged BANC-derived mapping ({} when absent)."""
    try:
        legs = json.loads(Path(path).read_text())["legs"]
    except (OSError, KeyError, ValueError):
        return {}
    out: dict[str, dict[str, list[int]]] = {}
    for side in ("left", "right"):
        per = {leg: sorted({r for a in legs.get(f"{leg}_{side}", {}).get("ans", ()) for r in a["flywire_root_ids"]}) for leg in ("front", "middle", "hind")}
        out[side] = {"front": per["front"], "all": sorted({r for v in per.values() for r in v})}
    return out


def leg_touch_rows(leg_rows: dict, sides: list[str]) -> np.ndarray:
    """Rows of the ascending neurons to drive: head-on (both sides) the front legs of both; else all legs of the contact side."""
    parts = [np.asarray(leg_rows.get(s, {}).get("front" if len(sides) > 1 else "all", ()), dtype=np.int64) for s in sides]
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)


def touch_sides(walls, yaw_deg: float) -> list[str]:
    """Antennae touched by the walls that blocked the fly: left / right relative to the heading, both when head-on.
    A wall behind the fly (more than 90 deg off the heading) touches no antenna."""
    sides: set[str] = set()
    for w in walls or ():
        rel = (_WALL_NORMAL_DEG[w] - float(yaw_deg) + 180.0) % 360.0 - 180.0
        if abs(rel) > 90.0:
            continue
        if abs(rel) <= TOUCH_HEAD_ON_DEG:
            sides.update(("left", "right"))
        else:
            sides.add("left" if rel > 0 else "right")
    return sorted(sides)


def _sides_from_rel(rel_deg: float) -> list[str]:
    """Antennae touched by a contact `rel_deg` from the heading (left +): both when head-on, none when behind."""
    if abs(rel_deg) > 90.0:
        return []
    if abs(rel_deg) <= TOUCH_HEAD_ON_DEG:
        return ["left", "right"]
    return ["left" if rel_deg > 0 else "right"]


MAX_COMBINED_HZ = 10000.0  # the brain's per-neuron ceiling (0.1 ms tick)


def arena_spec(interior_mm: float = ARENA_INTERIOR_MM) -> dict:
    h = float(interior_mm) / 2.0
    return {"kind": "glass_cube", "interior_mm": float(interior_mm), "bounds_mm": [-h, h, -h, h], "note": ARENA_NOTE}


# Giant-fibre takeoff trigger (OUR rule; calibrated on the stored exp-cl chunks, not validated): the combined DNp01 rate of a 50 ms
# chunk counts as a "hit" when > 0 Hz; takeoff fires when at least GF_MIN_HITS of the last GF_WINDOW_CHUNKS chunks are hits.
# 0 false alarms in 35 walking / rest runs (a, b, c, incl. blind and uniform-floor controls: at most 1 hit chunk in any 3);
# 10/10 loom runs (r/v 20 and 40 ms) trigger, mean 840 / 790 ms after the loom starts (the loom lasts 800 ms).
GF_WINDOW_CHUNKS = 3
GF_MIN_HITS = 2
GF_REFRACTORY_CHUNKS = 10  # no new takeoff for 500 ms after one finished
GF_RULE_NOTE = (f"takeoff trigger (ours, not validated): DNp01 > 0 Hz in >= {GF_MIN_HITS} of the last {GF_WINDOW_CHUNKS} chunks of 50 ms")

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
                 brain_lock=None, body_lock=None, arena: dict | None = None, vision=None,
                 taste_rows: dict[str, np.ndarray] | None = None, touch_rows: dict[str, np.ndarray] | None = None,
                 leg_rows: dict[str, dict[str, np.ndarray]] | None = None) -> None:
        self.brain, self.body, self.groups = brain, body, groups
        self.chunk_ms, self.mapper = float(chunk_ms), mapper or body_drive.DriveMapper(chunk_ms)
        self._clock, self.thermal, self._brain_fallback = clock, thermal, brain_fallback
        self.include_active, self.max_active, self.binary_frames = include_active, max_active, binary_frames
        # callables returning a fresh context manager per chunk (SimulationService.exclusive, a lock's `lambda: lock`)
        self._brain_lock, self._body_lock = brain_lock or contextlib.nullcontext, body_lock or contextlib.nullcontext
        self.arena = arena
        # playground world: sugar/coffee cubes; touching one drives its taste neurons (live_world, labelled and ours)
        self.world = live_world.World(arena["bounds_mm"] if arena else None)
        self.taste_rows = taste_rows or {}
        self._taste_sig: tuple = ()
        # antennal touch: sides blocked by a wall in the PREVIOUS chunk (the contact is known after the body moved)
        self.touch_rows = touch_rows or {}
        self.leg_rows = leg_rows or {}
        self._touch: list[str] = []
        self._touched: list[str] = []
        self._touch_latch: list[str] = []  # sides of the last surface-transition contact ...
        self._touch_left_ms = 0.0  # ... and how long it still fires
        self._quat: np.ndarray | None = None  # full 3D thorax orientation of the last chunk (w x y z), when the body reports it
        # the fly's eye (fly_eye.VisionFrontEnd or a stand-in): sets per-neuron T4/T5 rates every chunk from the pose of the last one
        self.vision = vision
        n0 = max(1, round(self.chunk_ms / 5.0))
        self._pose = (np.tile([0.0, 0.0, 1022.0], (n0, 1)), np.zeros(n0))  # the start pose, until the body reports one
        self._applied: tuple[np.ndarray, np.ndarray] | None = None
        self.seq, self.t_ms = 0, 0.0
        self.fallback: str | None = None
        # "kinematic" = recorded physics gait blended live, "physics" = the MuJoCo body stepped live (see body_kinematic)
        self.body_mode = getattr(body, "mode", "physics")
        self.body_caption = getattr(body, "caption", "")
        self._stimulus: tuple[list[int] | None, float] = (None, 0.0)
        self._speeds: deque[float] = deque(maxlen=SPEED_WINDOW)
        self._closed = False
        self._gf_hits: deque[bool] = deque(maxlen=GF_WINDOW_CHUNKS)
        self._land_hits: deque[bool] = deque(maxlen=GF_WINDOW_CHUNKS)
        self._gf_quiet = 0  # chunks left of the refractory period
        self._threat_az: dict[str, float] = {}  # threat id -> azimuth of its start from the heading, for the jump direction
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

    def world_add(self, item: str, ahead_mm: float = 3.0, left_mm: float = 0.0) -> dict:
        """Drop a cube `ahead_mm` in front of / `left_mm` to the left of the fly's current pose. ValueError on bad input."""
        with self._lock:
            pos, yaw = self._pose
            x, y, a = float(pos[-1][0]) / 1000.0, float(pos[-1][1]) / 1000.0, np.radians(float(yaw[-1]))
            wx = x + ahead_mm * np.cos(a) - left_mm * np.sin(a)
            wy = y + ahead_mm * np.sin(a) + left_mm * np.cos(a)
            return self.world.add(item, float(wx), float(wy))

    def world_threat(self, side: str = "front") -> dict:
        """Throw a dark sphere at the fly's head from `side` (left | right | front). ValueError on bad input."""
        with self._lock:
            pos, yaw = self._pose
            head = (float(pos[-1][0]) / 1000.0 + fly_eye.HEAD_FORWARD_MM * np.cos(np.radians(float(yaw[-1]))),
                    float(pos[-1][1]) / 1000.0 + fly_eye.HEAD_FORWARD_MM * np.sin(np.radians(float(yaw[-1]))),
                    float(pos[-1][2]) / 1000.0 + fly_eye.HEAD_UP_MM)
            obj = self.world.add_threat(side, head, float(yaw[-1]), self.t_ms)
            self._threat_az[obj["id"]] = live_world.THREAT_AZ_DEG[str(side).strip().lower()]
            return obj

    def _gf_trigger(self, dnp01_hz: float) -> float | None:
        """Update the GF window with this chunk's DNp01 rate; the jump direction (deg from the heading, away from the
        threat) when takeoff fires, else None."""
        if not hasattr(self.body, "takeoff") or getattr(self.body, "busy", False):
            self._gf_hits.clear()
            self._gf_quiet = GF_REFRACTORY_CHUNKS
            return None
        if self._gf_quiet > 0:
            self._gf_quiet -= 1
            return None
        self._gf_hits.append(dnp01_hz > 0.0)
        if sum(self._gf_hits) < GF_MIN_HITS:
            return None
        self._gf_hits.clear()
        azs = [self._threat_az[o["id"]] for o in self.world.objects if o["kind"] == "threat" and o["id"] in self._threat_az]
        return (azs[-1] + 180.0) if azs else 180.0

    def _land_trigger(self, landing_hz: float, airborne: bool) -> bool:
        """Landing rule (ours, not validated): DNp07 + DNp10 > 0 Hz in >= 2 of the last 3 chunks, counted only while airborne."""
        if not airborne:
            self._land_hits.clear()
            return False
        self._land_hits.append(landing_hz > 0.0)
        return sum(self._land_hits) >= GF_MIN_HITS

    def world_clear(self) -> int:
        with self._lock:
            return self.world.clear()

    def close(self) -> None:
        with self._lock:
            self._close()

    def _senses(self, extra: dict | None = None) -> dict:
        v = self.vision
        out = {"vision": VISION_LABEL if v is not None and getattr(v, "sees", True) else None,
               "background_hz": float(getattr(v, "background_hz", 0.0)) if v is not None else 0.0}
        note = getattr(v, "background_note", None)
        if note:
            out["background"] = note
        if v is not None:
            out.update(extra or {})
        return out

    def _apply_taste(self) -> None:
        """No eye: set user stimulus + taste rates on the brain when the touched set changed."""
        hz = self.world.taste(self._touched, self.taste_rows)
        sig = (tuple(sorted(hz.items())), tuple(self._touch))
        if sig == self._taste_sig:
            return
        self._taste_sig = sig
        if not hz and not self._touch:
            self._applied = None
            rows, rate = self._stimulus
            self.brain.set_stimulus(rows, rate)
            return
        self._apply_vision(None)

    def _apply_vision(self, out) -> None:
        """Combine the eye's rates, the user stimulus and the touched taste neurons (rates of the same neuron add)."""
        if out is None:
            rows, rates = np.zeros(0, dtype=np.int64), np.zeros(0)
        else:
            rows, rates = np.asarray(out.rows, dtype=np.int64), np.asarray(out.rates_hz, dtype=float)
        for preset, hz in self.world.taste(self._touched, self.taste_rows).items():
            tr = self.taste_rows[preset]
            rows, rates = np.concatenate([rows, tr]), np.concatenate([rates, np.full(len(tr), hz)])
        for side in self._touch:
            tr = self.touch_rows.get(side)
            if tr is not None and len(tr):
                rows, rates = np.concatenate([rows, tr]), np.concatenate([rates, np.full(len(tr), TOUCH_HZ)])
        lr = leg_touch_rows(self.leg_rows, self._touch)
        if len(lr):
            rows, rates = np.concatenate([rows, lr]), np.concatenate([rates, np.full(len(lr), LEG_TOUCH_HZ)])
        user_rows, user_rate = self._stimulus
        if user_rows and user_rate > 0:
            rows, rates = np.concatenate([rows, np.asarray(user_rows, dtype=np.int64)]), np.concatenate([rates, np.full(len(user_rows), user_rate)])
        uniq, inv = np.unique(rows, return_inverse=True)
        total = np.zeros(len(uniq))
        np.add.at(total, inv, rates)
        total = np.minimum(total, MAX_COMBINED_HZ)
        self._applied = (uniq, total)
        self.brain.set_rates(uniq, total)

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
                "dn_names": list(self.groups),
                "caption": body_drive.CAPTION + (" " + VISION_CAPTION if self._senses()["vision"] else ""),
                "senses": {**self._senses(), **({"caption": VISION_CAPTION} if self._senses()["vision"] else {})},
                "note": NOTE, "binary_frames": self.binary_frames, "arena": self.arena}

    # -- one chunk -----------------------------------------------------------------------

    def _to_cpu(self, why: str) -> None:
        old, self.brain = self.brain, self._brain_fallback()
        rows, rate = self._stimulus
        if self.vision is not None and self._applied is not None:
            self.brain.set_rates(*self._applied)
        elif rows:
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

    def _clamp_root(self, pos_um) -> dict:
        """Physics mode: clamp the reported root (in place) to the arena and say which wall stopped it."""
        xmin, xmax, ymin, ymax = (float(v) * 1000.0 for v in self.arena["bounds_mm"])
        pos = pos_um if isinstance(pos_um, np.ndarray) else np.asarray(pos_um)
        walls = []
        for col, lo, hi, names in ((0, xmin, xmax, ("x_min", "x_max")), (1, ymin, ymax, ("y_min", "y_max"))):
            if (pos[:, col] > hi).any():
                walls.append(names[1])
            if (pos[:, col] < lo).any():
                walls.append(names[0])
            pos[:, col] = np.clip(pos[:, col], lo, hi)
        return {"bounds_mm": list(self.arena["bounds_mm"]), "surface": "floor", "clamped_walls": sorted(walls),
                "note": PHYSICS_ARENA_NOTE}

    def _step_locked(self, force_binary: bool) -> tuple[dict, bytes | None]:
        if self._closed:
            raise RuntimeError("the live session is closed")
        t0 = self._clock()
        seen = None
        self._touched = self.world.touching(*self._pose)
        scene = getattr(self.vision, "scene", None)
        if scene is not None and hasattr(scene, "moving"):
            scene.moving, scene.t_ms = self.world.spheres(), self.t_ms
        if self.vision is not None:
            seen = self.vision.process(*self._pose, **({"quat": self._quat} if self._quat is not None else {}))
            self._apply_vision(seen)
            tb = self._clock()
        else:
            if self._touched or self._taste_sig or self._touch:
                with self._brain_lock():
                    self._apply_taste()
            tb = t0
        chunk = self._advance_brain()
        t1 = self._clock()
        dn = {name: chunk.group_hz(rows) for name, rows in self.groups.items()}
        sample = self.mapper.step(dn)
        away = self._gf_trigger(float(dn.get("dnp01", 0.0)))
        with self._body_lock():
            if away is not None:
                self.body.takeoff(away)
            if hasattr(self.body, "set_air") and getattr(self.body, "busy", False):
                self.body.set_air(dn.get("dng02", 0.0), dn.get("dnp03_left", 0.0), dn.get("dnp03_right", 0.0),
                                  self._land_trigger(float(dn.get("dnp07", 0.0)) + float(dn.get("dnp10", 0.0)), self.body.state == "airborne"))
            if getattr(self.body, "takes_command", False):  # kinematic body: the exact pace and turn, not a clipped drive
                out = self.body.advance(self.chunk_ms, sample.drive, sample.escape, pace=sample.pace, turn=sample.turn)
            else:
                out = self.body.advance(self.chunk_ms, sample.drive, sample.escape)
        t2 = self._clock()
        world = out.get("world")
        if self.arena and getattr(self.body, "bounds", None) is None:  # the body does not enforce walls: clamp what is reported
            out = {**out, "thorax": {**out["thorax"], "pos_um": np.array(out["thorax"]["pos_um"])}}
            world = self._clamp_root(out["thorax"]["pos_um"])
        self._pose = (np.asarray(out["thorax"]["pos_um"]), np.asarray(out["thorax"]["yaw_deg"]))
        touch_now = self._touch  # what this chunk's brain received
        q = out["thorax"].get("quat")
        self._quat = np.asarray(q) if q is not None else None
        if self.touch_rows:
            trans = (world or {}).get("transitions") or []
            if trans:  # brief contact at a surface transition: antennae on the side of the edge, both when head-on
                rel = float(trans[-1]["contact_rel_deg"])
                self._touch_latch = _sides_from_rel(rel)
                self._touch_left_ms = TOUCH_MS
            if self._touch_left_ms > 0:
                self._touch, self._touch_left_ms = list(self._touch_latch), self._touch_left_ms - self.chunk_ms
            else:
                self._touch = touch_sides((world or {}).get("clamped_walls"), float(self._pose[1][-1]))  # physics display clamp only
        else:
            self._touch = []
        wall_ms = max((t2 - t0) * 1000.0, 1e-6)
        factor = self.chunk_ms / wall_ms
        self._speeds.append(factor)
        self.t_ms += self.chunk_ms
        self.world.advance(self.t_ms)
        stim_rows, stim_rate = self._stimulus
        frames = np.asarray(out["frames"])
        binary = force_binary or self.binary_frames
        msg = {
            "type": "chunk", "seq": self.seq, "t_ms": round(self.t_ms, 3), "chunk_ms": self.chunk_ms,
            "speed_factor": round(factor, 4), "speed_factor_avg": round(float(np.mean(self._speeds)), 4),
            "speed_label": speed_label(factor),
            "wall_ms": {"brain": round((t1 - tb) * 1000.0, 3), "body": round((t2 - t1) * 1000.0, 3),
                        "total": round(wall_ms, 3)},
            "dn": {k: round(float(v), 2) for k, v in dn.items()},
            "drive": [round(float(v), 4) for v in out["drive"][-1]] if len(out["drive"]) else [0.0, 0.0],
            "forward": round(sample.forward, 4), "turn": round(sample.turn, 4), "escape": bool(sample.escape),
            "active_count": int(len(chunk.active_rows)), "backend": self.brain.backend, "fallback": self.fallback,
            "stimulus": {"rows": len(stim_rows), "rate_hz": stim_rate} if stim_rows and stim_rate > 0 else None,
            "frames": None if binary else frames.tolist(),
            "thorax": {"pos_um": np.asarray(out["thorax"]["pos_um"]).tolist(),
                       "yaw_deg": np.asarray(out["thorax"]["yaw_deg"]).tolist(),
                       **({"quat": np.asarray(out["thorax"]["quat"]).tolist()} if out["thorax"].get("quat") is not None else {})},
            "contact": np.asarray(out["contact"]).tolist(),
            "body_mode": self.body_mode, "body_caption": self.body_caption,
            "senses": self._senses({k: seen.stats[k] for k in ("driven_cells", "max_rate_hz", "mean_rate_hz")} if seen else None),
        }
        if "state" in out:
            msg["state"], msg["state_seq"] = out["state"], list(out["state_seq"])
            if "air" in out:
                msg["air"] = out["air"]
            if "wings" in out:
                msg["wings"], msg["wings_note"] = np.asarray(out["wings"]).tolist(), out["wings_note"]
                if "wing_cycle" in out:
                    msg["wing_cycle"] = np.asarray(out["wing_cycle"]).tolist()
            msg["takeoff"] = {"triggered": away is not None, "gf_hits": list(self._gf_hits), "rule": GF_RULE_NOTE, "note": getattr(self.body, "takeoff_note", None)}
        if self.touch_rows:
            msg["senses"]["touch"] = {"sides": touch_now, "driven_cells": sum(len(self.touch_rows.get(s, ())) for s in touch_now),
                                      "rate_hz": TOUCH_HZ if touch_now else 0.0, "note": TOUCH_NOTE,
                                      "leg_driven_cells": len(leg_touch_rows(self.leg_rows, touch_now)),
                                      "leg_rate_hz": LEG_TOUCH_HZ if touch_now and self.leg_rows else 0.0, "leg_note": LEG_TOUCH_NOTE}
        if seen is not None:
            msg["wall_ms"]["vision"] = round((tb - t0) * 1000.0, 3)
        if world or self.world.objects or self.world.used:
            msg["world"] = {**(world or {}), "objects": [dict(o) for o in self.world.objects], "touching": list(self._touched),
                            "taste_hz": self.world.taste(self._touched, self.taste_rows), "taste_note": live_world.NOTE}
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


VISION_MODES = (None, "arena")
MAX_BACKGROUND_HZ = 1000.0


def build_live_session(atlas, row_index, sims, body_lock=None, *, seed: int = 0, chunk_ms: float = CHUNK_MS,
                       include_active: bool = False, binary_frames: bool = False, body_mode: str = "kinematic",
                       vision: str | None = None, background_hz: float = 0.0) -> LiveSession:
    """The real brain (GPU when decided, usable and cool; else CPU) and the real body, ready to step.

    body_mode "kinematic" (default, fast: the recorded physics gait blended live, no flygym needed) or "physics"
    (MuJoCo stepped live, needs the `body` extra, slower). Raises BodyUnavailable (flygym or the motion library missing)
    or SimulationUnavailable (no weights) BEFORE anything heavy is built.

    vision "arena": the fly's eye (fly_eye.py) drives T4/T5 from its motion through the cube (needs the Codex column file,
    else fly_eye.VisionUnavailable). background_hz > 0 adds labelled Poisson noise on all mapped T4/T5 cells (works without
    vision: then the fly is blind but its motion detectors idle at that rate)."""
    from flybrain import body_runner
    from flybrain.sim_service import SimulationUnavailable

    arena = arena_spec()
    if vision not in VISION_MODES:
        raise ValueError(f"vision must be one of {VISION_MODES}")
    if not isinstance(background_hz, (int, float)) or isinstance(background_hz, bool) or not 0 <= background_hz <= MAX_BACKGROUND_HZ:
        raise ValueError(f"background_hz must be a number within 0..{MAX_BACKGROUND_HZ:g}")
    front = None
    if vision or background_hz > 0:
        eyes = fly_eye.load_eyes(atlas)
        if eyes is None:
            raise fly_eye.VisionUnavailable("vision needs the FlyWire column file data/raw/flywire_v783_column_assignment.csv.gz "
                                            "(Codex 'Visual Neuron Columns') and T4/T5 cells that it lists; it is missing or matches none")
        front = fly_eye.VisionFrontEnd(eyes, fly_eye.ArenaScene(interior_mm=arena["interior_mm"]), float(background_hz), see=vision == "arena")
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
                body = KinematicBody(load_library(), bounds=tuple(arena["bounds_mm"]), height_mm=arena["interior_mm"])
            except FileNotFoundError as exc:
                raise body_runner.BodyUnavailable(f"the motion library is missing ({exc}); run scripts/build_motion_library.py") from exc
        else:
            body = body_runner.BodyStream(seed=seed)
    except Exception:
        brain.close()
        raise
    cpu = sims.simulation()
    taste = {}
    for name in ("sugar", "bitter"):
        try:
            taste[name] = preset_rows(atlas, row_index, name)
        except Exception:  # noqa: BLE001 - no taste neurons in this atlas: touching a cube then drives nothing
            pass
    touch = {}
    try:
        nm = atlas.neurons
        for side in ("left", "right"):
            ct = nm["cell_type"].fillna("")
            sel = (nm["side"] == side) & (nm["cell_class"] == "mechanosensory") & (ct.str.startswith("JO") | ct.str.startswith("BM_Ant"))
            ids = nm.loc[sel, "root_id"].to_numpy()
            r = row_index.positions(ids)
            touch[side] = r[r >= 0].astype(np.int64)
    except Exception:  # noqa: BLE001 - no such cells in this atlas: touching a wall then drives nothing
        touch = {}
    leg = {}
    try:
        for side, d in load_leg_touch().items():
            leg[side] = {k: (lambda r: r[r >= 0].astype(np.int64))(row_index.positions(np.asarray(v, dtype=np.int64))) for k, v in d.items()}
    except Exception:  # noqa: BLE001 - no mapping / ids in this atlas: leg touch then drives nothing
        leg = {}
    return LiveSession(
        brain, body, readout_groups(atlas, row_index), chunk_ms=chunk_ms, thermal=sims.thermal,
        brain_fallback=lambda: cpu.open_stream(seed=seed), include_active=include_active,
        binary_frames=binary_frames, arena=arena, vision=front, taste_rows=taste, touch_rows=touch, leg_rows=leg, brain_lock=sims.exclusive, body_lock=(lambda: body_lock) if body_lock is not None else None,
    )
