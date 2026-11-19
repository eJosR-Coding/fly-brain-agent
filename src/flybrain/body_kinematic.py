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
* optional `bounds` (xmin, xmax, ymin, ymax in mm, world frame) + `height_mm`: the glass cube. SURFACE WALKING is a labelled
  BODY/WORLD RULE (adhesive pads), not the brain: the fly walks on one surface (floor, x_min, x_max, y_min, y_max, ceiling) with
  the same gait in the surface's local frame and crosses edges onto the next one (rule in body_surface.py); the chunk reports
  `world.surface`, the `world.transitions`, and the full 3D thorax orientation (`thorax.quat`, w x y z, arena frame).

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

from flybrain import body_drive, body_surface
from flybrain.body_runner import CPG_HZ, QUAT_SCALE, SAMPLE_HZ, chunk_frames

LIBRARY_PATH = Path(os.environ.get("FLYBRAIN_MOTION_LIBRARY") or Path(__file__).resolve().parents[2] / "web" / "public" / "fly-motion-library.json")
FORMAT = "fly-motion-library/1"
BINS = 50  # playback table: 3 bins per 5 ms frame at 12 Hz
RAMP_S = 0.1  # walking <-> standing transition
CAPTION = "Body: recorded physics gait (NeuroMechFly walking controller), blended live from the brain's drive; not a physics simulation."
WORLD_NOTE = ("World rule (not the brain): the fly's adhesive pads let it walk on any glass surface; at an edge it pitches onto the "
              "next surface (the heading keeps its tangential part, the into-wall part becomes up).")
FRAME_S = 1.0 / SAMPLE_HZ
FOOT_CLEARANCE_MM = 0.16  # the recorded leg tips sit up to 0.157 mm below the thorax-height reference plane: the thorax stands this much higher so no tip passes through the glass
# Gait phase rate follows the speed the body actually achieves (OUR choice, not a published map). Real flies step more
# slowly when they move less (DeAngelis et al. 2019, eLife 46409: stance duration ~ 1/forward speed; Wosnitza et al.
# 2013, J Exp Biol 216:480: speed correlates with cycle period); the library steps at a fixed 12 Hz, so a fly held
# by a wall would pedal at full rate on the spot (foot sliding). The phase rate is therefore 12 Hz x (achieved /
# commanded displacement), low-passed, floored so the legs never freeze while the walking command is on.
BLOCK_TAU_S = 0.15
RATE_FLOOR = 0.2


# Takeoff (short mode of the giant-fibre escape, von Reyn et al. 2014, Nat Neurosci 17:1 : GF-driven, middle legs extend,
# little wing raise) followed by FLIGHT replayed from real fly flight data (flybody flight-imitation dataset, Vaxenburg et al.
# 2025, Nature; GPL-3.0+; see data/fly-flight-clips.json). Everything numeric below is OUR labelled choice, not validated.
TAKEOFF_PREP_MS = 20.0  # middle legs extend; the root is still
TAKEOFF_LEG_EXTENSION = 1.35  # middle-leg bodies move to 1.35 x their distance from the coxa (pose only; quaternions untouched)
FLIGHT_PATH = Path(__file__).resolve().parent / "data" / "fly-flight-clips.json"
FLIGHT_SLOWDOWN = 3.0  # the recorded flight (0.2-0.6 s) plays 3x slower, so it lasts about a second on screen
FLIGHT_SCALE_MM = 5.0  # mm of world per dataset length unit (assumed cm: 10 mm) x 0.5, so the flight stays inside the 50 mm cube
FLIGHT_CEILING_MM = 45.0  # world rule: the glass ceiling
FLIGHT_TURN_MS = 150.0  # the body swings from its heading to the flight heading
RAMP_MS = 100.0  # wings unfold and legs tuck (and the reverse at landing)
LEG_TUCK = 0.55  # leg bodies move to 0.55 x their distance from the coxa in flight (a labelled flight pose)
DESCENT_S = 1.0  # after the clip: height eases to the floor, forward speed decays to zero
WING_DISPLAY_HZ = 10.0  # wingbeat pattern replayed at 10 Hz in simulated time (real ~200 Hz aliases at the 200 Hz frame rate)
WING_CYCLE_SAMPLES = 16  # one whole wingbeat per chunk (current weight and amplitude): the viewer flaps it at its own display rate
LANDED_MS = 100.0  # after touchdown the legs relax back to the standing pose and the wings fold
WING_NOTE = ("wings: flybody flight-imitation wingbeat (one real cycle: stroke, deviation, rotation; ~200 Hz in the data); "
             "both wings share it (no steering asymmetry yet)")
TAKEOFF_NOTE = ("takeoff + flight (ours, not validated): short-mode jump after the giant fibre rule fired, then a real fly flight clip "
                "(flybody dataset, Vaxenburg et al. 2025, GPL-3.0+) played "
                f"{FLIGHT_SLOWDOWN:g}x slower with the path scaled x{FLIGHT_SCALE_MM / 10:g} to fit the cube, a simple eased descent and a landing. "
                f"The wingbeat (~200 Hz in the data) is replayed at {WING_DISPLAY_HZ:g} Hz, a stated slowdown; legs tucked; no flight physics")
# Brain-driven flight (F3, ours, not validated). Rates are mean Hz per neuron of the DN group; at 0 Hz everything is the F2 pattern.
AIR_SMOOTH_S = 0.05  # first-order smoothing of the brain's flight command (a stated choice)
DNG02_REF_HZ, DNG02_GAIN = 50.0, 0.30  # stroke amplitude x (1 + 0.30 * min(rate / 50 Hz, 1)): direction from Namiki 2022, numbers ours
DNP03_REF_HZ, DNP03_AMP_GAIN = 50.0, 0.30  # asymmetry (L - R) / 50 Hz in [-1, 1]; ipsilateral stroke x (1 + 0.3 a), contralateral x (1 - 0.3 a): direction Croke 2025
DNP03_YAW_DEG_S = 90.0  # yaw rate = -90 deg/s * asymmetry (a left DNp03 turns the fly right: Croke's contralateral saccade); the size is ours
LAND_MIN_AIR_S = 0.5  # the landing DNs are ignored for the first 0.5 s of flight (the takeoff itself drives them); ours
MAX_AIR_S = 6.0  # max-airtime fallback: the fly lands after 6 s of flight even if the landing neurons never fire (ours)
AIR_NOTE = ("brain-driven flight (ours, not validated): DNg02 -> stroke amplitude (Namiki 2022), DNp03 left/right -> per-wing amplitude and "
            "yaw (Croke 2025), landing DNs DNp07/DNp10 -> landing (Croke 2025 names them landing neurons); every gain and threshold is ours")
MIDDLE_LEGS = ("lm", "rm")


@dataclass
class Flight:
    clips: list[dict]
    wing: np.ndarray  # (samples, 3) one wingbeat: stroke, deviation, rotation (rad)
    doc: dict


def load_flight(path: str | Path | None = None) -> Flight:
    doc = json.loads(Path(path or FLIGHT_PATH).read_text())
    if doc.get("format") != "fly-flight-clips/1":
        raise ValueError("not a fly-flight-clips/1 file")
    clips = [{**c, **{k: np.asarray(c[k], float) for k in ("x", "y", "z_abs", "yaw_deg")}} for c in doc["clips"]]
    return Flight(clips, np.asarray(doc["wing_pattern"]["rad"], float), doc)


def _smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


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
    takeoff_note = TAKEOFF_NOTE

    def __init__(self, library: MotionLibrary | None = None, bounds: tuple[float, float, float, float] | None = None,
                 flight: Flight | None = None, height_mm: float | None = None) -> None:
        self.lib = library or load_library()
        if bounds is not None:
            xmin, xmax, ymin, ymax = (float(v) for v in bounds)
            if not (xmin < xmax and ymin < ymax):
                raise ValueError("bounds must be (xmin, xmax, ymin, ymax) in mm with min < max")
            bounds = (xmin, xmax, ymin, ymax)
        self.bounds = bounds
        self.height_mm = float(height_mm) if height_mm is not None else (bounds[1] - bounds[0] if bounds else 1e9)
        # unbounded body: a huge box, so the same code path walks on the floor forever
        self._walker = body_surface.SurfaceWalker(bounds or (-1e9, 1e9, -1e9, 1e9), self.height_mm)
        self._z_base_um = 0.0
        self._z_sign = 1.0
        self.ticks = 0  # frames of walking so far
        self.cycles = 0.0  # gait phase in cycles: advances 12 Hz / 200 Hz per walking frame x rate_scale
        self.achieved = 1.0  # smoothed achieved / commanded root displacement (1 = free, 0 = blocked)
        self.frames = 0
        self.walk_w = 0.0  # 0 = standing, 1 = walking
        self.x_mm = self.y_mm = 0.0
        self._z_um = 0.0  # thorax height above the floor (um), arena frame
        self.yaw_rad = 0.0
        self._cmd: tuple[float, float] | None = None  # last (pace, turn)
        self.state = "standing"  # standing | walking | takeoff | airborne | landed
        self._to_t_ms: float | None = None  # ms since the takeoff trigger (None = no takeoff running)
        self._to_dir = (0.0, 0.0)
        self._land_ms = 0.0
        self._air_ms = 0.0  # time since lift-off (flight clip clock, simulated ms)
        self._desc_s = 0.0
        self._flight: Flight | None = flight
        self._clip: dict | None = None
        self._fl_head = self._fl_yaw0 = 0.0
        self._fl_prev = (0.0, 0.0)  # previous flight target (mm, heading frame x, y)
        self._fl_vel = (0.0, 0.0)  # last flight step (mm per frame, world frame)
        self._fl_lift_mm = 0.0
        self._wing_cycles = 0.0
        self._tuck = 0.0
        self._wing_w = 0.0
        self._air: dict | None = None  # the brain's flight command (set_air); None = F2 behaviour (clip end -> timed descent)
        self._air_f = {"amp": 1.0, "asym": 0.0}  # smoothed command
        self._yaw_steer = 0.0  # yaw (rad) accumulated by DNp03 steering in the air
        self.landing_cause: str | None = None
        lib = self.lib
        self._leg_idx = [i for i, b in enumerate(lib.bodies) if b[:2] in lib.legs]
        self._leg_cox = [next(j for j, b in enumerate(lib.bodies) if b == lib.bodies[i][:3] + "coxa") for i in self._leg_idx]
        self._leg_mid = np.array([lib.bodies[i][:2] in MIDDLE_LEGS for i in self._leg_idx])
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
        return self.cycles % 1.0

    @property
    def t_ms(self) -> float:
        return self.frames * FRAME_S * 1000.0

    def close(self) -> None:
        return None

    @property
    def busy(self) -> bool:
        """A takeoff is running (takeoff, airborne or landed)."""
        return self._to_t_ms is not None

    def takeoff(self, away_deg: float = 180.0, kind: str = "straight") -> bool:
        """Start the labelled takeoff + flight, heading `away_deg` from the current heading (180 = backwards). `kind` picks a
        flight clip: "straight", "left" or "right" (by the recorded net yaw). False if busy."""
        if self.busy:
            return False
        if self._flight is None:
            self._flight = load_flight()
        pool = [c for c in self._flight.clips if c["kind"] == kind] or self._flight.clips
        self._clip = pool[0]
        self._fl_head = self.yaw_rad + math.radians(away_deg)
        # takeoff from a wall or the ceiling (labelled): launch away from the surface normal; the body goes upright
        w = self._walker
        self._z_base_um = float(self._z_um or self.lib.stand.thorax_z_um)
        self._z_sign = 1.0
        if w.surface != body_surface.FLOOR:
            n = w.n
            if math.hypot(n[0], n[1]) > 0.5:
                self._fl_head = math.atan2(n[1], n[0])
            else:  # ceiling: away from it is down
                self._z_sign = -1.0
            w.start_free()
        self._to_dir = (math.cos(self._fl_head), math.sin(self._fl_head))
        self._to_t_ms, self._air_ms, self._land_ms, self._desc_s = 0.0, 0.0, 0.0, 0.0
        self._fl_prev, self._fl_vel, self._fl_lift_mm = (0.0, 0.0), (0.0, 0.0), 0.0
        self._wing_cycles = self._tuck = self._wing_w = 0.0
        self._air, self._air_f, self._yaw_steer, self.landing_cause = None, {"amp": 1.0, "asym": 0.0}, 0.0, None
        self.state = "takeoff"
        return True

    def set_air(self, dng02_hz: float, dnp03_left_hz: float, dnp03_right_hz: float, land: bool) -> None:
        """The brain's flight command for the next chunk (see the F3 constants): DNg02 sets the stroke amplitude, the DNp03 left - right
        asymmetry sets the per-wing amplitude and the yaw rate, `land` (the landing DNs crossed the live session's rule) lands the fly."""
        amp = 1.0 + DNG02_GAIN * min(max(dng02_hz / DNG02_REF_HZ, 0.0), 1.0)
        asym = min(max((dnp03_left_hz - dnp03_right_hz) / DNP03_REF_HZ, -1.0), 1.0)
        self._air = {"amp": amp, "asym": asym, "land": bool(land)}

    def _flight_at(self, t_s: float) -> tuple[float, float, float, float]:
        """(x, y mm in the heading frame, z mm above the floor, yaw rad relative) of the clip at recorded time t_s."""
        c = self._clip
        u = min(t_s * self._flight.doc["sample_rate_hz"], len(c["x"]) - 1.0)
        f = lambda k: float(np.interp(u, np.arange(len(c[k])), c[k]))
        return f("x") * FLIGHT_SCALE_MM, f("y") * FLIGHT_SCALE_MM, f("z_abs") * FLIGHT_SCALE_MM, math.radians(f("yaw_deg"))

    def _wing_angles(self, cycles: float | None = None) -> np.ndarray:
        """(6,) raw flybody wing angles in degrees [stroke, deviation, rotation] left then right, scaled by the wing weight."""
        pat = self._flight.wing
        u = ((self._wing_cycles if cycles is None else cycles) % 1.0) * len(pat)
        i = int(u) % len(pat)
        a = pat[i] + (pat[(i + 1) % len(pat)] - pat[i]) * (u - int(u))
        deg = np.degrees(a) * self._wing_w
        left, right = deg.copy(), deg.copy()
        if self._air is not None:  # stroke amplitude about the mean stroke: DNg02 (both wings), DNp03 asymmetry (opposite on the two wings)
            m = np.degrees(pat[:, 0].mean()) * self._wing_w
            amp, asym = self._air_f["amp"], self._air_f["asym"]
            left[0] = m + (deg[0] - m) * amp * (1.0 + DNP03_AMP_GAIN * asym)
            right[0] = m + (deg[0] - m) * amp * (1.0 - DNP03_AMP_GAIN * asym)
        return np.concatenate([left, right])

    def _takeoff_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, np.ndarray]:
        """One 5 ms frame of takeoff / flight / descent / landing: (pos, quat, contact, lift_um, dx_mm, dy_mm, wings).
        Advances the state machine and the yaw."""
        stand, ms = self.lib.stand, FRAME_S * 1000.0
        dx = dy = lift = 0.0
        ext = tuck = 0.0
        e_desc = None
        if self.state == "takeoff":
            ext = min(1.0, (self._to_t_ms + ms) / TAKEOFF_PREP_MS)
            self._to_t_ms += ms
            if self._to_t_ms >= TAKEOFF_PREP_MS:
                self.state, self._air_ms = "airborne", 0.0
                self._fl_yaw0 = self.yaw_rad
        elif self.state == "airborne":
            self._air_ms += ms
            t_rec = self._air_ms / 1000.0 / FLIGHT_SLOWDOWN
            x, y, z, yaw_rel = self._flight_at(t_rec)
            ramp = _smooth(self._air_ms / RAMP_MS)
            self._tuck = ramp
            self._wing_w = ramp
            turn = (self._fl_head - self._fl_yaw0 + math.pi) % (2 * math.pi) - math.pi
            if self._air is not None:
                k = min(1.0, FRAME_S / AIR_SMOOTH_S)
                for key in ("amp", "asym"):
                    self._air_f[key] += (self._air[key] - self._air_f[key]) * k
                self._yaw_steer += math.radians(-DNP03_YAW_DEG_S * self._air_f["asym"]) * FRAME_S
            self.yaw_rad = self._fl_yaw0 + turn * _smooth(self._air_ms / FLIGHT_TURN_MS) + yaw_rel + self._yaw_steer
            ch, sh = self._to_dir
            tx, ty = x * ch - y * sh, x * sh + y * ch
            dx, dy = tx - self._fl_prev[0], ty - self._fl_prev[1]
            self._fl_prev, self._fl_vel = (tx, ty), (dx, dy)
            self._fl_lift_mm = min(z * _smooth(self._air_ms / 100.0), FLIGHT_CEILING_MM)
            lift = self._fl_lift_mm * 1000.0
            ext = 1.0 - ramp
            if self._air is None:
                clip_end = t_rec >= (len(self._clip["x"]) - 1) / self._flight.doc["sample_rate_hz"]
                cause = "clip_end" if clip_end else None
            else:  # brain-driven: hover at the end of the clip until the landing DNs fire (or the max airtime)
                s_air = self._air_ms / 1000.0
                cause = ("landing_dn" if self._air["land"] and s_air >= LAND_MIN_AIR_S else "max_airtime" if s_air >= MAX_AIR_S else None)
            if cause:
                self.state, self._desc_s, self.landing_cause = "descending", 0.0, cause
        elif self.state == "descending":
            self._desc_s += FRAME_S
            u = self._desc_s / DESCENT_S
            e = _smooth(u)
            e_desc = e
            lift = self._fl_lift_mm * (1.0 - e) * 1000.0
            dx, dy = (v * (1.0 - e) for v in self._fl_vel)
            self._tuck = 1.0 - _smooth((self._desc_s - (DESCENT_S - 0.15)) / 0.15)  # legs come down before touchdown
            ext = 1.0 - self._tuck  # and the middle legs extend for it (reverse of the takeoff push)
            if u >= 1.0:
                self.state, self._land_ms, lift = "landed", 0.0, 0.0
        else:  # landed
            self._land_ms += ms
            r = max(0.0, 1.0 - self._land_ms / LANDED_MS)
            ext, self._tuck, self._wing_w = r, 0.0, r
            if self._land_ms >= LANDED_MS:
                self._to_t_ms, self.state = None, "standing"
        flying = self.state in ("airborne", "descending")
        hover = self._z_base_um + self._z_sign * self._fl_lift_mm * 1000.0
        if self.state in ("takeoff", "airborne") or (self.state == "descending" and e_desc is None):
            self._z_um = self._z_base_um + self._z_sign * lift
        elif self.state == "descending":
            self._z_um = hover * (1.0 - e_desc) + stand.thorax_z_um * e_desc  # always lands on the floor (labelled)
        else:
            self._z_um = float(stand.thorax_z_um)
        if self.bounds is not None:
            self._z_um = min(max(self._z_um, 300.0), (self.height_mm - body_surface.BODY_MARGIN_MM) * 1000.0)
        if flying or self._wing_w > 0.0:
            self._wing_cycles += WING_DISPLAY_HZ / SAMPLE_HZ
        tuck = self._tuck
        pos = stand.pos.copy()
        idx, cox = self._leg_idx, self._leg_cox
        scale = 1.0 + np.where(self._leg_mid, (TAKEOFF_LEG_EXTENSION - 1.0) * ext * (1.0 - tuck), 0.0) + (LEG_TUCK - 1.0) * tuck
        c = stand.pos[cox]
        pos[idx] = c + (stand.pos[idx] - c) * scale[:, None]
        contact = np.zeros_like(stand.contact) if flying else stand.contact
        return pos, stand.quat, contact, lift, dx, dy, self._wing_angles()

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
        wings = np.zeros((n, 6))
        tquat = np.zeros((n, 4))
        self._walker.events = []
        ev_frames: list[int] = []
        seq: list[str] = []
        for j in range(n):
            if self.busy:
                phase_now = self.state
                pos, quat, cont, lift, tdx, tdy, wing = self._takeoff_frame()
                wings[j] = wing
                nx, ny = self.x_mm + tdx, self.y_mm + tdy
                if self.bounds is not None:  # flight stays inside the cube
                    mg = body_surface.BODY_MARGIN_MM
                    nx, ny = min(max(nx, self.bounds[0] + mg), self.bounds[1] - mg), min(max(ny, self.bounds[2] + mg), self.bounds[3] - mg)
                self.x_mm, self.y_mm = nx, ny
                self.walk_w, self.achieved = 0.0, 1.0
                frames[j, :, :3] = np.round(pos)
                frames[j, :, 3:] = np.round(quat * QUAT_SCALE)
                contact[j] = cont
                _, tq = self._walker.render(0.0, body_surface.rz(self.yaw_rad), np.zeros(3))  # upright, eased from the surface frame
                thorax[j] = (round(nx * 1000.0), round(ny * 1000.0), round(self._z_um))
                tquat[j] = tq
                yaw[j] = math.degrees(self.yaw_rad)
                if not self.busy:  # landed: the fly stands on the floor again
                    self._walker.place_floor(self.x_mm, self.y_mm, self.yaw_rad)
                if not seq or seq[-1] != phase_now:
                    seq.append(phase_now)
                continue
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
            # root motion from the blended MEASURED speed / yaw rate on the CURRENT SURFACE (see body_surface.py)
            fwd, lat = speed * w * FRAME_S, lateral * w * FRAME_S
            n_ev = len(self._walker.events)
            self._walker.step(fwd, lat, math.radians(rate * w) * FRAME_S)
            ev_frames.extend([j] * (len(self._walker.events) - n_ev))
            self.achieved = 1.0  # pads never stall the legs: the gait keeps its full rate on any surface
            tp, tq = self._walker.render(z / 1000.0 + FOOT_CLEARANCE_MM)
            self.x_mm, self.y_mm, self.yaw_rad, self._z_um = float(tp[0]), float(tp[1]), self._walker.yaw, float(tp[2]) * 1000.0
            thorax[j] = np.round(tp * 1000.0)
            tquat[j] = tq
            yaw[j] = math.degrees(self.yaw_rad)
            if walking or w > 0.0:
                self.ticks += 1
                self.cycles += CPG_HZ / SAMPLE_HZ * max(RATE_FLOOR, self.achieved)
        self.frames += n
        if not self.busy:
            self.state = "walking" if walking and self.walk_w > 0.0 else "standing"
        out = {
            "state": self.state, "state_seq": seq or [self.state],
            "n_frames": n,
            "frames": frames,
            "contact": contact,
            "escape": np.full(n, int(escape)),
            "drive": np.tile(np.round(np.zeros(2) if escape else d, 4), (n, 1)),
            "thorax": {"pos_um": thorax, "yaw_deg": np.round(yaw, 3), "quat": np.round(tquat, 5)},
            "body_mode": self.mode,
        }
        if self._air is not None:
            out["air"] = {"amp": round(self._air_f["amp"], 3), "asym": round(self._air_f["asym"], 3), "land": self._air["land"],
                          "landing_cause": self.landing_cause, "note": AIR_NOTE}
        if wings.any():
            out["wings"] = np.round(wings, 2)
            out["wings_note"] = WING_NOTE
            out["wing_cycle"] = np.round([self._wing_angles(k / WING_CYCLE_SAMPLES) for k in range(WING_CYCLE_SAMPLES)], 2)
        w = self._walker
        t0 = self.frames - n
        trans = [{**e, "t_ms": round((t0 + j + 1) * FRAME_S * 1000.0, 3)} for e, j in zip(w.events, ev_frames)]
        out["world"] = {"bounds_mm": list(self.bounds) if self.bounds is not None else None,
                        "height_mm": self.height_mm if self.bounds is not None else None,
                        "surface": "flight" if self.busy and self.state in ("airborne", "descending") else w.surface,
                        "transitions": trans, "note": WORLD_NOTE if self.bounds is not None else None}
        return out
