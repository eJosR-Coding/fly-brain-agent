"""Motion front end (OUR model choice): a looming disc as local edge motion on the T4/T5 detectors.

T4 (ON edges) and T5 (OFF edges) come in four subtypes a-d, each tuned to one direction. Maisak et al. 2013
(Nature 500:212): a front-to-back, b back-to-front, c upward, d downward (lobula plate layers 1-4), in each eye.
Cells are placed by the same PCA retinotopy as eye.py, rotated into a head frame (posterior = +h, up = +v), so
both eyes share one convention. An expanding disc moves its edge radially outward, a receding one inward.
A cell in the edge band fires at base_rate * max(0, cos(angle between preferred direction and edge motion)),
quantised to LEVELS steps so a window is a few segments instead of one per cell.
"""

import numpy as np

from flybrain.eye import EyeMap, eye_map, window_edges
from flybrain.simulation import StimulusSchedule

PREFERRED = {"a": (1.0, 0.0), "b": (-1.0, 0.0), "c": (0.0, 1.0), "d": (0.0, -1.0)}  # (+h posterior, +v up)
MOTION_TYPES = tuple(f"T{g}{k}" for g in (4, 5) for k in "abcd")
POLARITIES = ("dark", "both")
LEVELS = 4  # cosine quantisation steps


def motion_eye(atlas, side: str = "right") -> tuple[EyeMap, np.ndarray]:
    """Head-frame map of one side's T4a-d/T5a-d and each row's cell type."""
    emap = eye_map(atlas, side, MOTION_TYPES).head_frame()
    return emap, atlas.neurons["cell_type"].to_numpy()[emap.rows]


def motion_schedule(emap: EyeMap, types, radii, duration_ms: float, rate_hz: float, direction: int = 1,
                    polarity: str = "dark") -> StimulusSchedule:
    """Edge motion of a disc whose radius reaches `radii[i]` at the end of window i.

    direction 1 = expanding (edge moves outward through the band r[i-1]..r[i]), -1 = receding (the same bands
    in reverse order, edge moving inward). polarity "dark": a dark disc, so an expanding edge is OFF (T5) and a
    shrinking edge is ON (T4); "both": T4 and T5 alike.
    """
    if polarity not in POLARITIES:
        raise ValueError(f"polarity must be one of {POLARITIES}, got {polarity!r}")
    if direction not in (1, -1):
        raise ValueError(f"direction must be 1 (expanding) or -1 (receding), got {direction!r}")
    types = np.asarray(types)
    if polarity == "dark":
        allowed = np.array([t.startswith("T5" if direction == 1 else "T4") for t in types])
    else:
        allowed = np.ones(len(types), dtype=bool)
    pref = np.array([PREFERRED[t[-1]] for t in types])
    dist = np.linalg.norm(emap.xy, axis=1)
    radial = emap.xy / np.where(dist > 0, dist, 1.0)[:, None]
    weight = np.maximum(0.0, (pref * radial).sum(axis=1) * direction)
    level = np.round(weight * LEVELS) / LEVELS
    bounds = [0.0, *radii]
    bands = list(zip(bounds, bounds[1:]))
    if direction == -1:
        bands = bands[::-1]
    edges = window_edges(duration_ms, len(radii))
    segments = []
    for (start, end), (lo, hi) in zip(zip(edges, edges[1:]), bands):
        in_band = (dist > lo) & (dist <= hi) if lo > 0 else dist <= hi
        for lv in sorted(set(level[in_band & allowed & (level > 0)])):
            rows = emap.rows[in_band & allowed & (level == lv)]
            segments.append((start, end, rows, rate_hz * float(lv)))
    if not segments:
        raise ValueError("no motion drives any T4/T5 cell: the edge bands miss every cell tuned to them")
    return StimulusSchedule(segments)
