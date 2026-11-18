"""Eye map: a flat, retinotopic view of one eye's photoreceptors, and looming stimulus schedules.

R1-6 positions in FlyWire sit in the lamina/medulla, which is retinotopic, so squashing one eye's
positions onto their two main axes (PCA) gives a rough picture of the visual field. Orientation and
mirroring do not matter for a disc, only who is near whom does.
"""

import math

import numpy as np

from flybrain.simulation import StimulusSchedule

PHOTORECEPTOR_TYPE = "R1-6"
SIDES = ("left", "right")

# Looming geometry: an object of half-size l approaching at speed v subtends 2*atan(l / (v * ttc))
# at time-to-contact ttc. Only l/v matters, so one constant carries it.
LOOM_L_OVER_V_MS = 40.0  # l/v in ms: a mid-range fly escape stimulus (10-100 ms in the literature)
COLLISION_LEAD_MS = 5.0  # collision happens this long after the run ends, so the end is huge but finite
FULL_FIELD_DEG = 180.0  # visual angle that fills the whole eye disc (radius fraction 1)


class EyeMap:
    """Photoreceptor rows with 2D coordinates normalised so the farthest one sits at radius 1."""

    def __init__(self, rows, positions) -> None:
        pos = np.asarray(positions, dtype=np.float64)
        rows = np.asarray(rows, dtype=np.int64)
        if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) != len(rows):
            raise ValueError("positions must be (n, 3) with one row id per point")
        if len(rows) < 3:
            raise ValueError(f"an eye map needs at least 3 photoreceptors, got {len(rows)}")
        centred = pos - pos.mean(axis=0)
        _, _, axes = np.linalg.svd(centred, full_matrices=False)
        flat = centred @ axes[:2].T
        extent = float(np.linalg.norm(flat, axis=1).max())
        if extent <= 0:
            raise ValueError("photoreceptor positions are all identical: no eye shape to map")
        self.rows = rows
        self.xy = flat / extent  # centroid at (0, 0), farthest point at radius 1
        self._axes, self._centred = axes[:2], centred  # kept so head_frame() can orient the map

    def __len__(self) -> int:
        return len(self.rows)

    def head_frame(self) -> "EyeMap":
        """The same map rotated into the head frame: +x toward posterior, +y upward (both eyes alike).

        FlyWire axes (checked on the atlas): y grows ventrally, z grows posteriorly, x toward the fly's right.
        The map's vertical is the in-plane projection of dorsal (-y); its horizontal is perpendicular to that,
        signed so that it correlates positively with z (posterior). Our choice: the flattening is only approximate.
        """
        up = self._axes @ np.array([0.0, -1.0, 0.0])
        if np.linalg.norm(up) < 0.2:
            raise ValueError("the map plane has no dorsoventral extent: cannot orient it to the head")
        v_dir = up / np.linalg.norm(up)
        h_dir = np.array([-v_dir[1], v_dir[0]])
        z = self._centred[:, 2]
        if float(np.dot(self.xy @ h_dir, z)) < 0:
            h_dir = -h_dir
        out = EyeMap.__new__(EyeMap)
        out.rows, out._axes, out._centred = self.rows, self._axes, self._centred
        out.xy = np.column_stack([self.xy @ h_dir, self.xy @ v_dir])
        return out

    def disc(self, radius: float, centre=None) -> np.ndarray:
        """Rows (sorted) whose map position is within `radius` (fraction of the eye radius) of `centre`.

        `centre` is (x, y) in map units, default the centroid. An empty disc gives an empty array.
        """
        if not math.isfinite(radius) or radius < 0:
            raise ValueError(f"radius must be a finite number >= 0, got {radius!r}")
        c = np.zeros(2) if centre is None else np.asarray(centre, dtype=np.float64)
        if c.shape != (2,) or not np.isfinite(c).all():
            raise ValueError(f"centre must be two finite numbers (x, y), got {centre!r}")
        inside = np.linalg.norm(self.xy - c, axis=1) <= radius
        return np.sort(self.rows[inside])


def eye_map(atlas, side: str = "right", cell_type: str | tuple[str, ...] = PHOTORECEPTOR_TYPE) -> EyeMap:
    """The EyeMap of one side's cells of `cell_type` (one name or several) from the atlas ("left" or "right")."""
    if side not in SIDES:
        raise ValueError(f'side must be "left" or "right", got {side!r}')
    n = atlas.neurons
    types = (cell_type,) if isinstance(cell_type, str) else tuple(cell_type)
    rows = np.flatnonzero((n["cell_type"].isin(types) & (n["side"] == side)).to_numpy())
    if not len(rows):
        raise ValueError(f"no {'/'.join(types)} neurons on the {side} side in the atlas")
    return EyeMap(rows, n[["x_um", "y_um", "z_um"]].to_numpy()[rows])


def looming_radii(duration_ms: float, steps: int, l_over_v_ms: float = LOOM_L_OVER_V_MS,
                  lead_ms: float = COLLISION_LEAD_MS, full_field_deg: float = FULL_FIELD_DEG) -> list[float]:
    """Disc radius (fraction of the eye radius) at the end of each of `steps` equal windows.

    Angular size of the approaching object: theta = 2 * atan(l_over_v / ttc), with ttc the time left
    to collision (collision at duration + lead). Radius fraction = theta / full_field, capped at 1.
    Slow at first, then accelerating: the signature of an object on a collision course.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    out = []
    for i in range(1, steps + 1):
        ttc = duration_ms + lead_ms - duration_ms * i / steps
        theta = math.degrees(2 * math.atan(l_over_v_ms / ttc))
        out.append(min(1.0, theta / full_field_deg))
    return out


def window_edges(duration_ms: float, steps: int) -> list[float]:
    """steps + 1 edges, rounded to the simulator's 0.1 ms grid."""
    return [round(duration_ms * i / steps, 1) for i in range(steps + 1)]


def disc_schedule(eye: EyeMap, radii, duration_ms: float, rate_hz: float, centre=None) -> StimulusSchedule:
    """One segment per window: the photoreceptors inside the disc of that window's radius fire at `rate_hz`.

    Windows with an empty disc are left dark. At least one window needs a non-empty disc.
    """
    edges = window_edges(duration_ms, len(radii))
    segments = []
    for (start, end), radius in zip(zip(edges, edges[1:]), radii):
        rows = eye.disc(radius, centre)
        if len(rows):
            segments.append((start, end, rows, rate_hz))
    if not segments:
        raise ValueError("every window's disc is empty: no photoreceptor would be stimulated")
    return StimulusSchedule(segments)


def schedule_drive(schedule: StimulusSchedule) -> float:
    """Total input drive: sum over neurons of rate (Hz) x active time (s) = expected input events."""
    return sum(len(rows) * rate * (end - start) / 1000.0 for start, end, rows, rate in schedule.segments)
