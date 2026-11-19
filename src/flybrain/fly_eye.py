"""What the fly SEES inside the arena, as T4/T5 input for the live brain (CL1). Pure numpy.

Pipeline, once per live chunk (50 ms = ten 200 Hz frames of the fly's pose):

1. Eye geometry. Every T4/T5 cell sits in a medulla column (p, q) of the exact FlyWire column map (retina.py: lattice ->
   h posterior+, v up+ in radians from the eye axis, a fit of the measured lens map). A column's unit view direction in
   the head frame (x forward, y left, z up) puts the eye axis at EYE_AXIS_AZIMUTH_DEG from the heading, to the side of
   its eye (OUR choice: the column file gives angles from the eye axis, not its head-frame pose), mirror image for the
   other eye. Head = body for now (no head pitch or roll).
2. Scene (`ArenaScene`): rays from the head through the glass cube. The floor carries a soft checker of a stated period,
   through the glass the background is light, the 12 cube edges are dark lines, optional objects are dark boxes. Receptor
   blur (acceptance angle) is modelled analytically as range-dependent attenuation: far checkers wash out, far thin
   lines widen and fade. All our modelling choices, none measured.
3. Motion (`Hrc`): a Hassenstein-Reichardt correlator per column along each subtype's preferred direction (Maisak et al.
   2013: a front-to-back, b back-to-front, c up, d down). Weber contrast against a slow adaptation state, half-wave
   split into ON (T4: brightening) and OFF (T5: darkening) channels, first-order low-pass delay on the neighbour column
   UPSTREAM of the preferred direction, full opponent, rectified. A static scene or a uniform luminance change gives
   exactly zero. The photoreceptor / lamina / medulla path is BYPASSED: this replaces it, it is not a model of it.
4. Rate: drive in [0, 1] -> Poisson rate = MAX_RATE_HZ * drive. MAX_RATE_HZ = 100 Hz is the CALIBRATED scale (see the comment at
   MAX_RATE_HZ: loom fires the ipsilateral giant fiber, floor flow, receding and dimming do not, for 75-200 Hz); the saturating
   map and its gain are OUR choice. The detector's temporal optimum is 3 Hz (HRC_OPTIMAL_TF_HZ, tau 53 ms), our choice inside the
   published bracket.

The eye sees the pose of the chunk that just ended and its rates drive the NEXT chunk: one chunk (50 ms) of latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from flybrain.motion import PREFERRED
from flybrain.retina import column_angles

SUBTYPES = "abcd"
EYE_AXIS_AZIMUTH_DEG = 65.0  # eye axis from the heading, toward its own side (our choice)
HEAD_FORWARD_MM, HEAD_UP_MM = 0.5, 0.3  # head relative to the thorax (our choice)
FRAME_MS = 5.0  # 200 Hz frames, as the body reports
HRC_OPTIMAL_TF_HZ = 3.0  # temporal-frequency optimum of the detector, our choice inside the measured bracket (see TAU_LP_MS)
# Reichardt delay (low-pass) time constant: a first-order delay peaks at f = 1 / (2 pi tau). Bracket (calcium imaging, Strother et al.
# 2018 PNAS, PMC5776785): a resting T4 gives monotonically smaller responses from 1 to 27 Hz gratings, behavioural state
# (walking) lifts them mostly between 1 and 10 Hz; no direct measurement of the optimum for a walking fly was found, so 3 Hz is OUR choice.
TAU_LP_MS = 1000.0 / (2.0 * np.pi * HRC_OPTIMAL_TF_HZ)  # 53.05 ms
TAU_ADAPT_MS = 300.0  # slow adaptation: contrast is relative to this running mean (our choice)
HR0 = 0.02  # drive = 1 - exp(-mean_hr / HR0): saturating map from correlator output to [0, 1] (our choice)
# THE one free scale of the eye (T4/T5 Poisson rate at drive 1). No published firing rate of T4/T5 exists: whole-cell recordings find graded
# potentials (no verified spikes; Gruntman et al. 2018 Nat Neurosci PMC5967973, Gruntman et al. 2019 eLife PMC6917495), so this is a stand-in for
# the synaptic output of a graded cell. CALIBRATED against the live brain (odd/tasks/playground.md, CL1 calibration; 12 brain seeds per cell,
# detector tau 53 ms): 50 Hz is the first scale at which a dark looming disc (r/v 20 and 40 ms, right eye) fires the ipsilateral DNp01 (0.83 of the
# trials), from 75 to 200 Hz it does in all 12 trials while receding, a uniform dimming and walking at 14 mm/s over the checkered floor fire
# neither DNp01 in any trial; at 300 Hz the floor flow fires DNp01 (0.75). 100 Hz sits in the middle of that range (geometric centre ~120 Hz).
# The window is a property of THIS front end and weights (e.g. the connectome has no inhibition that cancels floor flow), not a measurement.
MAX_RATE_HZ = 100.0
MAX_RATE_CALIBRATED = True
MIN_RATE_HZ = 1.0  # driven cells below this are left out (a handful of near-zero rows per chunk are not worth sending)
NEIGHBOUR_COS = 0.8  # the upstream column must lie within ~37 deg of the opposite of the preferred direction
# the six nearest columns in (p, q): from the fitted lens map the steps (+-1, 0), (0, +-1), +-(1, 1) are each ~4 deg away
# (the (1, -1) step is ~8 deg, the second ring)
LATTICE_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1))
MOTION_TYPES = tuple(f"T{g}{k}" for g in (4, 5) for k in SUBTYPES)
VISION_LABEL = "arena (T4/T5 via HRC; photoreceptor path bypassed)"

# scene (our choices)
FLOOR_MEAN, FLOOR_CONTRAST, FLOOR_SHARPNESS = 0.6, 0.3, 3.0
BACKGROUND = 1.0
EDGE_WIDTH_MM, EDGE_DARKNESS = 1.5, 0.8
ACCEPTANCE_SIGMA_RAD = np.radians(2.1)  # sigma of a ~5 deg FWHM receptor acceptance angle


class VisionUnavailable(RuntimeError):
    """The eye cannot be built (the Codex column file is missing or matches no T4/T5 cell)."""


@dataclass
class FlyEye:
    """One eye: the columns (unique (p, q)), their head-frame directions, the upstream neighbours and the T4/T5 cells."""

    side: str
    p: np.ndarray
    q: np.ndarray
    h: np.ndarray  # radians, posterior +
    v: np.ndarray  # radians, up +
    dirs: np.ndarray  # (n_columns, 3) unit view directions, head frame (x forward, y left, z up)
    upstream: np.ndarray  # (4, 2, n_columns) int: the columns upstream of each subtype's preferred direction, -1 if none
    cell_rows: np.ndarray  # atlas rows of the cells
    cell_types: np.ndarray  # 'T4a' .. 'T5d'
    cell_col: np.ndarray  # column index of each cell
    cell_sub: np.ndarray  # 0..3 (a..d)
    cell_on: np.ndarray  # True for T4 (ON), False for T5 (OFF)

    def cell_drive(self, on: np.ndarray, off: np.ndarray) -> np.ndarray:
        """Per-cell drive from the per-column (4, n_columns) ON / OFF drives."""
        return np.where(self.cell_on, on[self.cell_sub, self.cell_col], off[self.cell_sub, self.cell_col])


def _directions(side: str, h: np.ndarray, v: np.ndarray) -> np.ndarray:
    az = np.radians(EYE_AXIS_AZIMUTH_DEG) + h  # toward the eye's own side, growing toward the back
    sign = -1.0 if side == "right" else 1.0
    return np.stack([np.cos(v) * np.cos(az), sign * np.cos(v) * np.sin(az), np.sin(v)], axis=-1)


def _upstream(p, q, h, v) -> np.ndarray:
    """(4, 2, n) upstream columns per subtype: the best neighbour on each side of the direction opposite to the preferred
    one (within NEIGHBOUR_COS), -1 where there is none. Two flanking neighbours cancel the cross-talk of orthogonal
    motion, because the lattice axes are not aligned with a/b/c/d (the horizontal ones are ~25 deg off)."""
    key = p.astype(np.int64) * 100003 + q.astype(np.int64)
    order = np.argsort(key)
    skey = key[order]
    out = np.full((4, 2, len(p)), -1, dtype=np.int64)
    best = np.full((4, 2, len(p)), -1.0)
    for dp, dq in LATTICE_STEPS:
        want = (p + dp).astype(np.int64) * 100003 + (q + dq)
        pos = np.clip(np.searchsorted(skey, want), 0, len(p) - 1)
        found = skey[pos] == want
        nb = order[pos]
        dh, dv = h[nb] - h, v[nb] - v
        norm = np.maximum(np.hypot(dh, dv), 1e-12)
        for s, sub in enumerate(SUBTYPES):
            t = -np.asarray(PREFERRED[sub])  # the upstream direction
            cos = (dh * t[0] + dv * t[1]) / norm
            side = (t[0] * dv - t[1] * dh) < 0  # which flank of the upstream direction the neighbour lies on
            for k in (0, 1):
                better = found & (cos >= NEIGHBOUR_COS) & (side == bool(k)) & (cos > best[s, k])
                out[s, k][better], best[s, k][better] = nb[better], cos[better]
    return out


def build_eye(side: str, cell_rows, cell_types, cell_p, cell_q) -> FlyEye:
    """An eye from its T4/T5 cells: atlas rows, types ('T4a'..'T5d') and lattice coordinates (p, q) of each cell."""
    if side not in ("left", "right"):
        raise ValueError(f'side must be "left" or "right", got {side!r}')
    pq = np.column_stack([np.asarray(cell_p, dtype=np.int64), np.asarray(cell_q, dtype=np.int64)])
    cols, inverse = np.unique(pq, axis=0, return_inverse=True)
    p, q = cols[:, 0], cols[:, 1]
    h, v = column_angles(p, q)
    types = np.asarray(cell_types)
    return FlyEye(
        side, p, q, h, v, _directions(side, h, v), _upstream(p, q, h, v),
        np.asarray(cell_rows, dtype=np.int64), types, inverse.reshape(-1).astype(np.int64),
        np.array([SUBTYPES.index(t[-1]) for t in types], dtype=np.int64), np.array([t.startswith("T4") for t in types]),
    )


def build_eye_from_columns(side: str, p, q, first_row: int = 0) -> FlyEye:
    """A full eye on the given lattice columns: all eight T4/T5 types in every column (tests, synthetic eyes)."""
    p, q = np.asarray(p), np.asarray(q)
    n = len(p)
    types = np.repeat(np.array(MOTION_TYPES)[None, :], n, axis=0).reshape(-1)
    return build_eye(side, first_row + np.arange(8 * n), types, np.repeat(p, 8), np.repeat(q, 8))


def load_eyes(atlas, path=None) -> dict[str, FlyEye] | None:
    """{side: FlyEye} from the atlas and the Codex column file, or None when the file is missing or matches nothing.

    Only T4/T5 cells the column file lists are in the eye; the others stay undriven (count in `eye_info`)."""
    import pandas as pd

    from flybrain import retina

    path = retina.COLUMN_FILE if path is None else path
    try:
        cols = pd.read_csv(path, usecols=["root_id", "p", "q"]).drop_duplicates("root_id")
    except (OSError, ValueError, EOFError, pd.errors.ParserError):
        return None
    n = atlas.neurons
    eyes = {}
    for side in ("right", "left"):
        cells = n[(n["side"] == side) & n["cell_type"].isin(MOTION_TYPES)]
        hit = pd.DataFrame({"root_id": cells["root_id"].to_numpy(), "row": cells.index.to_numpy(),
                            "type": cells["cell_type"].to_numpy()}).merge(cols, on="root_id")
        if len(hit) >= 3:
            eyes[side] = build_eye(side, hit["row"].to_numpy(), hit["type"].to_numpy(), hit["p"].to_numpy(), hit["q"].to_numpy())
    return eyes or None


# -- the scene ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Box:
    """A dark axis-aligned box (mm, arena frame: x forward at the start, y left, z up from the floor)."""

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    lum: float = 0.1


@dataclass(frozen=True)
class MovingSphere:
    """A dark sphere flying on a straight line at constant velocity (mm, arena frame). Alive for `life_ms` after `t0_ms`."""

    p0: tuple[float, float, float]
    vel_mm_s: tuple[float, float, float]
    radius_mm: float
    t0_ms: float
    life_ms: float = float("inf")
    stop_ms: float = float("inf")  # it stops moving this long after the spawn (and stays until life_ms)


LOOM_EDGE_DEG = 1.0  # soft edge of a disc, same as the calibration sweep's looming disc
LOOM_DARKENING = 0.9  # luminance x (1 - 0.9 * mask) inside the disc, same as the calibration


class ArenaScene:
    """The glass cube (interior_mm wide, centred on the start, floor at z = 0) as luminance in [0, 1] along rays."""

    def __init__(self, interior_mm: float = 50.0, floor_period_mm: float = 4.0, objects=(),
                 edge_width_mm: float = EDGE_WIDTH_MM) -> None:
        self.half = float(interior_mm) / 2.0
        self.height = float(interior_mm)
        self.period, self.objects, self.edge_width = float(floor_period_mm), tuple(objects), float(edge_width_mm)
        self.background = BACKGROUND
        self._lo = np.array([-self.half, -self.half, 0.0])
        self._hi = np.array([self.half, self.half, self.height])
        self.moving: list[MovingSphere] = []  # set per chunk by the caller together with `t_ms` (the chunk's start time)
        self.t_ms = 0.0

    def luminance(self, eye_pos_mm, yaw_deg, dirs_head) -> np.ndarray:
        """(F, n) luminance seen from eye positions (F, 3) mm with headings yaw_deg (F,) along head-frame dirs (n, 3)."""
        pos = np.asarray(eye_pos_mm, dtype=np.float64)
        c, s = np.cos(np.radians(np.asarray(yaw_deg, dtype=np.float64))), np.sin(np.radians(np.asarray(yaw_deg, dtype=np.float64)))
        d = np.asarray(dirs_head, dtype=np.float64)
        dx = c[:, None] * d[None, :, 0] - s[:, None] * d[None, :, 1]
        dy = s[:, None] * d[None, :, 0] + c[:, None] * d[None, :, 1]
        dz = np.broadcast_to(d[None, :, 2], dx.shape)
        dw = np.stack([dx, dy, dz], axis=-1)  # (F, n, 3) world-frame directions
        o = np.clip(pos, self._lo + 1e-6, self._hi - 1e-6)[:, None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            bound = np.where(dw > 0, self._hi, self._lo)
            t_axis = np.where(dw != 0, (bound - o) / dw, np.inf)
        face = np.argmin(t_axis, axis=-1)
        t = np.take_along_axis(t_axis, face[..., None], axis=-1)[..., 0]
        hit = o + t[..., None] * dw
        sigma = t * ACCEPTANCE_SIGMA_RAD  # receptor blur in mm at the exit point

        lum = np.full(t.shape, self.background)
        on_floor = (face == 2) & (dw[..., 2] < 0)
        if on_floor.any():
            k = 2.0 * np.pi / self.period
            quarter = self.period / 4.0  # the start sits at the middle of a square
            c2 = np.sin(k * (hit[..., 0] + quarter)) * np.sin(k * (hit[..., 1] + quarter))
            atten = np.exp(-((k * sigma) ** 2))
            floor = FLOOR_MEAN + FLOOR_CONTRAST * np.clip(FLOOR_SHARPNESS * c2, -1.0, 1.0) * atten
            lum = np.where(on_floor, floor, lum)
        # edges: distance on the exit face to the nearest of its bordering edges
        dist = np.minimum(hit - self._lo, self._hi - hit)  # (F, n, 3) distance to each pair of faces
        dist = np.where(np.arange(3) == face[..., None], np.inf, dist)
        edge_d = dist.min(axis=-1)
        sigma_line = self.edge_width / 2.0
        sigma_tot = np.sqrt(sigma_line**2 + sigma**2)
        amp = EDGE_DARKNESS * sigma_line / sigma_tot
        lum = lum * (1.0 - amp * np.exp(-0.5 * (edge_d / sigma_tot) ** 2))
        for box in self.objects:
            lo, hi = np.asarray(box.lo, float), np.asarray(box.hi, float)
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo - o) / dw, (hi - o) / dw
            inside_slab = (dw == 0) & (o >= lo) & (o <= hi)
            tmin = np.where(dw == 0, np.where(inside_slab, -np.inf, np.inf), np.minimum(t1, t2)).max(axis=-1)
            tmax = np.where(dw == 0, np.where(inside_slab, np.inf, -np.inf), np.maximum(t1, t2)).min(axis=-1)
            lum = np.where((tmax >= np.maximum(tmin, 0.0)) & (np.maximum(tmin, 0.0) < t), box.lum, lum)
        for sph in self.moving:
            lum = self._sphere(lum, o, dw, t, sph)
        return lum

    def _sphere(self, lum, o, dw, t, sph: MovingSphere) -> np.ndarray:
        """Darken the rays that fall inside the angular disc of a moving sphere (the calibration's soft-edged disc model:
        half-angle asin(r / d), 1 deg sigmoid edge, luminance x (1 - 0.9 mask)); frame f is at t_ms + 5 ms (f + 1)."""
        F = dw.shape[0]
        te = self.t_ms + FRAME_MS * (np.arange(F) + 1.0) - sph.t0_ms  # (F,) ms since spawn
        alive = (te >= 0.0) & (te <= sph.life_ms)
        if not alive.any():
            return lum
        c = np.asarray(sph.p0, float)[None, :] + np.asarray(sph.vel_mm_s, float)[None, :] * (np.minimum(te, sph.stop_ms)[:, None] / 1000.0)  # (F, 3)
        oc = c[:, None, :] - o  # (F, 1, 3) eye -> centre
        d = np.linalg.norm(oc, axis=-1)  # (F, 1)
        u = oc / np.maximum(d, 1e-9)[..., None]
        ang = np.arccos(np.clip((dw * u).sum(axis=-1), -1.0, 1.0))  # (F, n)
        theta = np.arcsin(np.clip(sph.radius_mm / np.maximum(d, 1e-9), 0.0, 1.0))  # (F, 1) half-angle
        mask = 1.0 / (1.0 + np.exp((ang - theta) / np.radians(LOOM_EDGE_DEG)))
        mask = mask * alive[:, None] * (d < t)  # not behind the glass
        return lum * (1.0 - LOOM_DARKENING * mask)


# -- motion ----------------------------------------------------------------------------------------------------

@dataclass
class Drive:
    on: np.ndarray  # (4, n_columns) mean drive over the frames, T4 channel per subtype a..d
    off: np.ndarray  # (4, n_columns) T5 channel


class Hrc:
    """Reichardt correlators of one eye with their state (adaptation and delay low-passes) carried across chunks."""

    def __init__(self, eye: FlyEye, dt_ms: float = FRAME_MS, tau_lp_ms: float = TAU_LP_MS, tau_adapt_ms: float = TAU_ADAPT_MS) -> None:
        self.eye, self.a_lp, self.a_ad = eye, dt_ms / (tau_lp_ms + dt_ms), dt_ms / (tau_adapt_ms + dt_ms)
        self._valid = eye.upstream >= 0  # (4, 2, n)
        self._up = np.where(self._valid, eye.upstream, 0)
        self._count = np.maximum(self._valid.sum(axis=1), 1)  # flanking neighbours available per subtype and column
        self._mean = self._y_on = self._y_off = None

    def step(self, lum: np.ndarray) -> Drive:
        """Advance over the (F, n_columns) luminance frames and return the drive averaged over them."""
        lum = np.asarray(lum, dtype=np.float64)
        n = lum.shape[1]
        if self._mean is None:
            self._mean = lum[0].copy()
            self._y_on, self._y_off = np.zeros(n), np.zeros(n)
        acc_on, acc_off = np.zeros((4, n)), np.zeros((4, n))
        for frame in lum:
            contrast = np.clip((frame - self._mean) / np.maximum(self._mean, 0.05), -1.0, 1.0)
            x_on, x_off = np.maximum(contrast, 0.0), np.maximum(-contrast, 0.0)
            self._y_on += self.a_lp * (x_on - self._y_on)
            self._y_off += self.a_lp * (x_off - self._y_off)
            self._mean += self.a_ad * (frame - self._mean)
            for x, y, acc in ((x_on, self._y_on, acc_on), (x_off, self._y_off, acc_off)):
                hr = y[self._up] * x[None, None, :] - y[None, None, :] * x[self._up]  # delayed upstream * direct - mirror
                acc += np.maximum((np.where(self._valid, hr, 0.0)).sum(axis=1) / self._count, 0.0)
        f = float(len(lum))
        return Drive(1.0 - np.exp(-acc_on / f / HR0), 1.0 - np.exp(-acc_off / f / HR0))


# -- the front end ---------------------------------------------------------------------------------------------

@dataclass
class VisionOut:
    rows: np.ndarray  # atlas rows to drive (unique)
    rates_hz: np.ndarray
    stats: dict


class VisionFrontEnd:
    """Pose of the last chunk in, T4/T5 rows and Poisson rates for the next chunk out (both eyes)."""

    def __init__(self, eyes: dict[str, FlyEye], scene: ArenaScene | None = None, background_hz: float = 0.0,
                 see: bool = True) -> None:
        """`see=False` keeps only the labelled background noise (no scene, no motion): the fly is blind but its T4/T5 idle."""
        if background_hz < 0 or not np.isfinite(background_hz):
            raise ValueError("background_hz must be a finite number >= 0")
        self.eyes, self.scene, self.background_hz, self.sees = eyes, scene or ArenaScene(), float(background_hz), bool(see)
        self._hrc = {side: Hrc(eye) for side, eye in eyes.items()}
        self._rows = np.concatenate([e.cell_rows for e in eyes.values()])
        self.n_cells = int(len(self._rows))
        self.background_note = (
            f"background noise: {self.background_hz:g} Hz Poisson on all {self.n_cells} mapped T4/T5 cells (both eyes), "
            "added to the visual drive; our choice (a spontaneous-activity floor), not a measured rate"
        ) if self.background_hz > 0 else None

    def process(self, pos_um, yaw_deg) -> VisionOut:
        t0 = time.perf_counter()
        pos = np.asarray(pos_um, dtype=np.float64) / 1000.0
        yaw = np.asarray(yaw_deg, dtype=np.float64)
        rad = np.radians(yaw)
        eye_pos = np.column_stack([pos[:, 0] + HEAD_FORWARD_MM * np.cos(rad), pos[:, 1] + HEAD_FORWARD_MM * np.sin(rad),
                                   pos[:, 2] + HEAD_UP_MM])
        rows, rates = [], []
        for side, eye in self.eyes.items():
            rows.append(eye.cell_rows)
            if self.sees:
                drive = self._hrc[side].step(self.scene.luminance(eye_pos, yaw, eye.dirs))
                rates.append(MAX_RATE_HZ * eye.cell_drive(drive.on, drive.off))
            else:
                rates.append(np.zeros(len(eye.cell_rows)))
        rows, vis = np.concatenate(rows), np.concatenate(rates)
        driven = vis >= MIN_RATE_HZ
        vis = np.where(driven, vis, 0.0)
        total = vis + self.background_hz
        keep = total > 0
        order = np.argsort(rows[keep], kind="stable")
        out_rows, out_rates = rows[keep][order], total[keep][order]
        stats = {"driven_cells": int(driven.sum()), "max_rate_hz": round(float(vis.max()), 2) if len(vis) else 0.0,
                 "mean_rate_hz": round(float(vis[driven].mean()), 2) if driven.any() else 0.0,
                 "ms": round((time.perf_counter() - t0) * 1000.0, 3)}
        return VisionOut(out_rows, out_rates, stats)
