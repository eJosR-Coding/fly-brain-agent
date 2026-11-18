"""Kinematic live body: a recorded physics gait, blended live (pure numpy; no flygym import).

`scripts/build_motion_library.py` records NeuroMechFly's walking controller (physics) over a grid of walking commands
(pace x turn) into web/public/fly-motion-library.json: one 12 Hz gait cycle per command as thorax-relative leg-body
transforms, with the MEASURED thorax forward speed, lateral speed and yaw rate of that run, phase-aligned (bin 0 = LF
swing onset). `KinematicBody` replays that library with the BodyStream interface the live session uses:

* a continuously advancing gait phase (12 Hz x t, exactly 3 bins of the 50-bin cycle per 5 ms frame), shared by all clips;
* the pose is the weighted blend, at that equal phase, of the 1-4 library clips around the command (bilinear weights on
  the (pace, turn) grid: normalised, continuous, exactly one clip on a grid point): positions lerp, quaternions nlerp;
* the command moves linearly from the previous chunk's value to this chunk's across the chunk, so nothing pops;
* standing (zero drive) and escape ramp the walking weight to zero over 100 ms: the pose slides to the recorded standing
  pose, the root stops, escape also raises the `escape` flag (the viewer raises the wings; no takeoff model);
* root motion integrates the BLENDED measured speed and yaw rate (and lateral slip), so the fly goes anywhere and curves.
  Positive turn = left = positive yaw, as in body_drive / body_runner.
* optional `bounds` (xmin, xmax, ymin, ymax in mm, world frame): a labelled WORLD RULE, not the brain: the root is clamped
  to the box (a wall stops forward motion; a slanted approach slides along it) and the chunk says `world.blocked`.

Honest limits: this is a replay of recorded gaits, not a physics simulation (no terrain contact, no obstacle forces,
the legs do not react to the wall); `advance(drive=...)` without an explicit command inverts the unclipped
body_drive.walking_drive, which is exact except where a side was clipped to the drive limits (the live session passes the
exact pace and turn). Deterministic.
"""

from __future__ import annotations

import base64
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from flybrain import body_drive
from flybrain.body_runner import CPG_HZ, QUAT_SCALE, SAMPLE_HZ, chunk_frames

LIBRARY_PATH = Path(os.environ.get("FLYBRAIN_MOTION_LIBRARY") or Path(__file__).resolve().parents[2] / "web" / "public" / "fly-motion-library.json")
FORMAT = "fly-motion-library/1"
BINS = 50  # playback table: 3 bins per 5 ms frame at 12 Hz
RAMP_S = 0.1  # walking <-> standing transition
CAPTION = "Body: recorded physics gait (NeuroMechFly walking controller), blended live from the brain's drive; not a physics simulation."
WORLD_NOTE = "World rule (not the brain): the glass wall stops the fly's root motion; the legs keep stepping."
FRAME_S = 1.0 / SAMPLE_HZ


def _unpack(b64: str, shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype="<i2").reshape(shape).astype(float)


def _cubic_periodic(a: np.ndarray, factor: int) -> np.ndarray:
    """Catmull-Rom upsample of the (periodic) first axis; same rule as scripts/build_motion_library.py."""
    n = len(a)
    i = np.arange(n * factor) // factor
    t = ((np.arange(n * factor) % factor) / factor).reshape((-1,) + (1,) * (a.ndim - 1))
    p0, p1, p2, p3 = a[(i - 1) % n], a[i % n], a[(i + 1) % n], a[(i + 2) % n]
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)


@dataclass
class Clip:
    id: str
    kind: str
    pace: float
    turn: float
    drive: tuple[float, float]
    speed_mm_s: float
    lateral_mm_s: float
    yaw_rate_deg_s: float
    thorax_z_um: float
    pos: np.ndarray  # (phase, 48, 3) um
    quat: np.ndarray  # (phase, 48, 4)
    contact: np.ndarray  # (phase, 6) bool
    seam: dict = field(default_factory=dict)
    table_pos: np.ndarray = field(default=None, repr=False)  # (BINS, 48, 3) playback table
    table_quat: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        factor = BINS // len(self.pos)
        self.table_pos = _cubic_periodic(self.pos, factor)
        q = _cubic_periodic(self.quat, factor)
        self.table_quat = q / np.linalg.norm(q, axis=-1, keepdims=True)


@dataclass
class Stand:
    pos: np.ndarray
    quat: np.ndarray
    contact: np.ndarray
    thorax_z_um: float


class MotionLibrary:
    def __init__(self, doc: dict) -> None:
        if doc.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} document (format {doc.get('format')!r})")
        self.doc = doc
        self.bodies, self.legs = list(doc["bodies"]), list(doc["legs"])
        self.phase_samples = int(doc["phase_samples"])
        self.pace_grid = tuple(float(v) for v in doc["pace_grid"])
        self.turn_grid = tuple(float(v) for v in doc["turn_grid"])
        self.neutral_pos = np.array(doc["neutral"]["pos_um"], float)
        self.neutral_quat = np.array(doc["neutral"]["quat"], float) / doc["quat_scale"]
        scale, nb, nl = float(doc["quat_scale"]), len(self.bodies), len(self.legs)

        def decode(b64, n):
            f = _unpack(b64, (n, nb, 7))
            return f[..., :3], f[..., 3:] / scale

        def bits(mask, n):
            return np.array([[(int(m) >> i) & 1 for i in range(nl)] for m in mask], bool).reshape(n, nl)

        self.clips: list[Clip] = []
        for c in doc["clips"]:
            pos, quat = decode(c["frames_b64"], self.phase_samples)
            self.clips.append(Clip(c["id"], c["kind"], float(c["pace"]), float(c["turn"]), tuple(c["drive"]),
                                   float(c["speed_mm_s"]), float(c["lateral_mm_s"]), float(c["yaw_rate_deg_s"]),
                                   float(c["thorax_z_um"]), pos, quat, bits(c["contact"], self.phase_samples), c.get("seam", {})))
        self._by_id = {c.id: c for c in self.clips}
        sp, sq = decode(doc["stand"]["frames_b64"], 1)
        self.stand = Stand(sp[0], sq[0], bits(doc["stand"]["contact"], 1)[0], float(doc["stand"]["thorax_z_um"]))
        self._walk = {(c.pace, c.turn): c for c in self.clips if c.kind == "walk"}
        if set(self._walk) != {(p, t) for p in self.pace_grid for t in self.turn_grid}:
            raise ValueError("the library does not cover the whole (pace, turn) grid")

    def clip_by_id(self, cid: str) -> Clip:
        return self._by_id[cid]

    def clip_at(self, pace: float, turn: float) -> Clip:
        """The walking clip at an exact grid point."""
        return self._walk[(float(pace), float(turn))]

    def weights(self, pace: float, turn: float) -> dict[str, float]:
        """Bilinear weights of the 1-4 walking clips around (pace, turn), clamped to the grid; they sum to 1 and an
        exact grid point returns that clip alone."""
        if not (math.isfinite(pace) and math.isfinite(turn)):
            raise ValueError("pace and turn must be finite")
        (i0, i1, fp), (j0, j1, ft) = (_cell(self.pace_grid, pace), _cell(self.turn_grid, turn))
        out: dict[str, float] = {}
        for i, wp in ((i0, 1.0 - fp), (i1, fp)):
            for j, wt in ((j0, 1.0 - ft), (j1, ft)):
                w = wp * wt
                if w > 1e-12:
                    cid = self._walk[(self.pace_grid[i], self.turn_grid[j])].id
                    out[cid] = out.get(cid, 0.0) + w
        total = sum(out.values())
        return {k: v / total for k, v in out.items()}


def _cell(grid: tuple[float, ...], x: float) -> tuple[int, int, float]:
    x = min(max(x, grid[0]), grid[-1])
    i = min(max(int(np.searchsorted(grid, x, side="right")) - 1, 0), len(grid) - 2)
    f = (x - grid[i]) / (grid[i + 1] - grid[i])
    if f < 1e-9:
        f = 0.0
    elif f > 1.0 - 1e-9:
        f = 1.0
    return i, i + 1, f


_CACHE: dict[Path, MotionLibrary] = {}


def load_library(path: str | Path | None = None) -> MotionLibrary:
    p = Path(path) if path is not None else LIBRARY_PATH
    key = p.resolve()
    if path is None and key in _CACHE:
        return _CACHE[key]
    lib = MotionLibrary(json.loads(p.read_text()))
    if path is None:
        _CACHE[key] = lib
    return lib


class KinematicBody:
    """The live body of the kinematic mode: see the module doc. Not thread-safe (the caller serialises chunks)."""

    mode = "kinematic"
    takes_command = True  # advance() accepts the exact pace / turn (the live session passes them)
    caption = CAPTION

    def __init__(self, library: MotionLibrary | None = None, bounds: tuple[float, float, float, float] | None = None) -> None:
        self.lib = library or load_library()
        if bounds is not None:
            xmin, xmax, ymin, ymax = (float(v) for v in bounds)
            if not (xmin < xmax and ymin < ymax):
                raise ValueError("bounds must be (xmin, xmax, ymin, ymax) in mm with min < max")
            bounds = (xmin, xmax, ymin, ymax)
        self.bounds = bounds
        self.ticks = 0  # frames of walking so far: the gait phase is ticks x 12 Hz / 200 Hz
        self.frames = 0
        self.walk_w = 0.0  # 0 = standing, 1 = walking
        self.x_mm = self.y_mm = 0.0
        self.yaw_rad = 0.0
        self._cmd: tuple[float, float] | None = None  # last (pace, turn)
        lib = self.lib
        self.header = {
            "format": "fly-walk/1",
            "units": {"position": "um", "quaternion": f"w,x,y,z x {QUAT_SCALE}", "time": "s"},
            "frame": "fly frame relative to the thorax body: x anterior, y left, z dorsal; rotation = body frame -> thorax frame",
            "quat_scale": QUAT_SCALE,
            "thorax_origin_um": list(lib.doc["thorax_origin_um"]),
            "bodies": lib.bodies,
            "legs": lib.legs,
            "sample_rate_hz": SAMPLE_HZ,
            "cycle_hz": CPG_HZ,
            "neutral": lib.doc["neutral"],
            "frame_layout": "frames[i][body] = [x, y, z (um), qw, qx, qy, qz (x quat_scale)]",
            "body_mode": self.mode,
            "body_caption": CAPTION,
        }

    @property
    def phase(self) -> float:
        return (self.ticks * CPG_HZ / SAMPLE_HZ) % 1.0

    @property
    def t_ms(self) -> float:
        return self.frames * FRAME_S * 1000.0

    def close(self) -> None:
        return None

    # -- command ------------------------------------------------------------------------

    @staticmethod
    def _command(drive: np.ndarray, pace, turn) -> tuple[float, float, bool]:
        """(pace, turn, walking) of a chunk. An explicit pace/turn wins; else the drive is inverted (see module doc)."""
        if pace is not None or turn is not None:
            p = 0.0 if pace is None else float(pace)
            t = 0.0 if turn is None else float(turn)
            if not (math.isfinite(p) and math.isfinite(t)):
                raise ValueError("pace and turn must be finite")
            return min(max(p, 0.0), 1.0), min(max(t, -1.0), 1.0), p > 0.0
        base = float(drive.mean())
        if base <= 1e-9:
            return 0.0, 0.0, False
        p = (base - body_drive.DRIVE_MIN) / (body_drive.BASE_MAX - body_drive.DRIVE_MIN)
        t = (drive[1] - drive[0]) / (2.0 * body_drive.TURN_GAIN * base)
        return min(max(p, 0.0), 1.0), min(max(float(t), -1.0), 1.0), True

    # -- one chunk ----------------------------------------------------------------------

    def advance(self, ms: float, drive, escape: bool = False, *, pace: float | None = None, turn: float | None = None) -> dict:
        n = chunk_frames(ms)
        d = np.asarray(drive, dtype=float).reshape(-1)
        if d.shape != (2,) or not np.isfinite(d).all():
            raise ValueError(f"drive must be two finite numbers [left, right], got {drive!r}")
        escape = bool(escape)
        c_pace, c_turn, walking = self._command(d, pace, turn)
        walking = walking and not escape
        prev = self._cmd if self._cmd is not None else (c_pace, c_turn)
        self._cmd = (c_pace, c_turn)
        lib, stand = self.lib, self.lib.stand
        step_w = FRAME_S / RAMP_S
        frames = np.empty((n, len(lib.bodies), 7), int)
        contact = np.empty((n, len(lib.legs)), int)
        thorax = np.empty((n, 3), int)
        yaw = np.empty(n)
        blocked = False
        for j in range(n):
            a = (j + 1) / n
            p, t = prev[0] + (c_pace - prev[0]) * a, prev[1] + (c_turn - prev[1]) * a
            self.walk_w = min(1.0, self.walk_w + step_w) if walking else max(0.0, self.walk_w - step_w)
            w = self.walk_w
            weights = lib.weights(p, t)
            phase = self.phase
            x = phase * BINS
            i0 = int(math.floor(x + 1e-9)) % BINS
            f = max(0.0, x - math.floor(x + 1e-9))
            i1 = (i0 + 1) % BINS
            pos = np.zeros_like(stand.pos)
            quat = np.zeros_like(stand.quat)
            speed = lateral = rate = z = 0.0
            lead, lead_w = None, -1.0
            for cid, wt in weights.items():
                c = lib.clip_by_id(cid)
                pos += wt * ((1.0 - f) * c.table_pos[i0] + f * c.table_pos[i1])
                quat += wt * ((1.0 - f) * c.table_quat[i0] + f * c.table_quat[i1])
                speed += wt * c.speed_mm_s; lateral += wt * c.lateral_mm_s; rate += wt * c.yaw_rate_deg_s; z += wt * c.thorax_z_um
                if wt > lead_w:
                    lead, lead_w = c, wt
            pos = w * pos + (1.0 - w) * stand.pos
            quat = w * quat + (1.0 - w) * stand.quat
            quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
            z = w * z + (1.0 - w) * stand.thorax_z_um
            frames[j, :, :3] = np.round(pos)
            frames[j, :, 3:] = np.round(quat * QUAT_SCALE)
            contact[j] = (lead.contact[int(round(x)) % BINS * len(lead.contact) // BINS] if w >= 0.5 else stand.contact)
            # root motion from the blended MEASURED speed / yaw rate, scaled by the walking weight
            mid = self.yaw_rad + math.radians(rate * w) * FRAME_S * 0.5
            fwd, lat = speed * w * FRAME_S, lateral * w * FRAME_S
            nx = self.x_mm + fwd * math.cos(mid) - lat * math.sin(mid)
            ny = self.y_mm + fwd * math.sin(mid) + lat * math.cos(mid)
            if self.bounds is not None:
                cx, cy = min(max(nx, self.bounds[0]), self.bounds[1]), min(max(ny, self.bounds[2]), self.bounds[3])
                blocked = blocked or cx != nx or cy != ny
                nx, ny = cx, cy
            self.x_mm, self.y_mm = nx, ny
            self.yaw_rad += math.radians(rate * w) * FRAME_S
            thorax[j] = (round(nx * 1000.0), round(ny * 1000.0), round(z))
            yaw[j] = math.degrees(self.yaw_rad)
            if walking or w > 0.0:
                self.ticks += 1
        self.frames += n
        out = {
            "n_frames": n,
            "frames": frames,
            "contact": contact,
            "escape": np.full(n, int(escape)),
            "drive": np.tile(np.round(np.zeros(2) if escape else d, 4), (n, 1)),
            "thorax": {"pos_um": thorax, "yaw_deg": np.round(yaw, 3)},
            "body_mode": self.mode,
        }
        if self.bounds is not None:
            out["world"] = {"bounds_mm": list(self.bounds), "blocked": blocked, "note": WORLD_NOTE}
        else:
            out["world"] = {"bounds_mm": None, "blocked": False, "note": None}
        return out
