"""Exact retinotopy of T4/T5 (V2): where each motion detector looks, from the FlyWire Codex column file.

The Codex "Visual Neuron Columns" table (Matsliah et al. 2024) gives every T4/T5 a medulla column (p, q) on the hex
lattice of Zhao et al. 2025's eye map. Hex axes: y = p + q is dorsal (up), x = (q - p) / 2 is frontal, in both eyes.
A column's visual direction is NOT a linear function of (p, q): the lens map bends (a plain lattice scaled by the
mean spacing differs from the measured directions by ~20 deg median). So h (posterior +) and v (up +), in radians
from the eye axis, come from a cubic polynomial in (p / 20, q / 20) fitted to the 776 measured lens directions of
Zhao et al. (median error 1.0 deg, 90th percentile 2.0 deg on all T4/T5 columns; columns outside the measured
lattice are extrapolated by the same polynomial). Only 20 fitted coefficients live here, no per-column data; the
column file itself (data/raw/, gitignored) is read at run time. Our model choice; preferred directions stay the
canonical a/b/c/d of Maisak et al. 2013 (motion.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from flybrain.eye import EyeMap

ROOT = Path(__file__).resolve().parents[2]
COLUMN_FILE = ROOT / "data" / "raw" / "flywire_v783_column_assignment.csv.gz"
MOTION_TYPES = tuple(f"T{g}{k}" for g in (4, 5) for k in "abcd")
SCALE = 20.0
DEGREE = 3
# monomials p^i q^j for i in 0..3, j in 0..3-i (same order as _features), one coefficient row per output
_POWERS = [(i, j) for i in range(DEGREE + 1) for j in range(DEGREE + 1 - i)]
_COEF_H = (-0.0206133, -1.34874, 0.364781, -0.522949, 1.40307, -0.5174, 0.846865, 0.219293, -0.625316, 0.339925)
_COEF_V = (-0.018371, 0.783341, -0.170582, 0.262473, 0.630837, -0.0452196, 0.0668034, 0.15499, 0.154205, 0.248545)


def column_angles(p, q) -> tuple[np.ndarray, np.ndarray]:
    """(h, v) in radians (h posterior +, v up +) of lattice columns (p, q), from the fitted lens map."""
    p, q = np.asarray(p, dtype=np.float64) / SCALE, np.asarray(q, dtype=np.float64) / SCALE
    feats = np.stack([p**i * q**j for i, j in _POWERS], axis=-1)
    return feats @ np.asarray(_COEF_H), feats @ np.asarray(_COEF_V)


def exact_motion_eye(atlas, side: str, path: Path | str = COLUMN_FILE):
    """(EyeMap in the head frame, cell types, {"undriven", "columns", "source"}) for one side, or None.

    None when the column file is missing/unreadable or matches fewer than 3 of the side's T4/T5 cells.
    T4/T5 cells of the side that the file does not list stay out of the map (undriven) and are counted.
    Positions are scaled so the farthest column sits at radius 1, like EyeMap.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        cols = pd.read_csv(path, usecols=["root_id", "p", "q"]).drop_duplicates("root_id")
    except (OSError, ValueError, pd.errors.ParserError, EOFError):
        return None
    n = atlas.neurons
    cells = n[(n["side"] == side) & n["cell_type"].isin(MOTION_TYPES)]
    pos = pd.DataFrame({"root_id": cells["root_id"].to_numpy(), "row": cells.index.to_numpy()})
    hit = pos.merge(cols, on="root_id")
    if len(hit) < 3:
        return None
    h, v = column_angles(hit["p"].to_numpy(), hit["q"].to_numpy())
    xy = np.column_stack([h, v])
    extent = float(np.linalg.norm(xy, axis=1).max())
    if extent <= 0:
        return None
    rows = hit["row"].to_numpy(dtype=np.int64)  # the atlas index is a RangeIndex: label == row
    emap = EyeMap.from_xy(rows, xy / extent)
    info = {"undriven": int(len(cells) - len(hit)), "columns": int(len(hit)), "extent_deg": round(float(np.degrees(extent)), 1)}
    return emap, n["cell_type"].to_numpy()[rows], info
