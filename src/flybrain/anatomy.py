"""Anatomy context derived from the data: region codes per neuron and group landmarks.

Nothing here is hardcoded geometry: every number is a centroid, a direction or a percentile
of the real atlas positions. Missing groups are omitted, never faked.
"""

import numpy as np

from flybrain.atlas import Atlas

REGION_CODES = {0: "central", 1: "optic", 2: "sensory", 3: "other"}
_OTHER = 3
NT_CODES = {0: "acetylcholine", 1: "glutamate", 2: "gaba", 3: "dopamine", 4: "serotonin", 5: "octopamine", 6: "unknown"}
_NT_UNKNOWN = 6
_NT_TO_CODE = {name: code for code, name in NT_CODES.items() if code != _NT_UNKNOWN}
_SUPER_CLASS_TO_CODE = {
    "central": 0,
    "optic": 1, "visual_projection": 1, "visual_centrifugal": 1,
    "sensory": 2, "sensory_ascending": 2,
}
# Lamina and photoreceptor terminals (R1-6 are annotated "sensory"): the outer bulge of each optic lobe.
LAMINA_PATTERN = r"^(?:L[1-5]|L1-3|R1-6|R7|R8|Lawf[12])$"
_SIDES = (("left", "L"), ("right", "R"))
LOBE_RADIUS_PERCENTILE = 80


def region_codes(atlas: Atlas) -> np.ndarray:
    """One uint8 per atlas row (same order as /brain/points); see REGION_CODES."""
    mapped = atlas.neurons["super_class"].map(_SUPER_CLASS_TO_CODE).fillna(_OTHER)
    return mapped.to_numpy().astype(np.uint8)


def nt_codes(atlas: Atlas) -> np.ndarray:
    """One uint8 per atlas row (same order as /brain/points): the predicted top neurotransmitter, see NT_CODES."""
    names = atlas.neurons["top_nt"].astype("string").str.strip().str.lower()
    return names.map(_NT_TO_CODE).fillna(_NT_UNKNOWN).to_numpy().astype(np.uint8)


def _xyz(df) -> np.ndarray:
    return df[["x_um", "y_um", "z_um"]].to_numpy(dtype="float64")


def _landmark(name: str, pts: np.ndarray) -> dict:
    c = pts.mean(axis=0)
    return {"name": name, "x": float(c[0]), "y": float(c[1]), "z": float(c[2]), "count": int(len(pts))}


def landmarks(atlas: Atlas) -> dict:
    """Centroids (um) per group plus, per side, the lamina direction and a robust lobe radius."""
    n = atlas.neurons
    sc = n["super_class"]
    out: list[dict] = []
    sides: dict[str, dict] = {}

    central = n[sc == "central"]
    if len(central):
        out.append(_landmark("central brain", _xyz(central)))

    is_lamina = n["cell_type"].str.match(LAMINA_PATTERN).fillna(False).astype(bool)
    for side, letter in _SIDES:
        on_side = n["side"] == side
        lobe = _xyz(n[(sc == "optic") & on_side])
        if not len(lobe):
            continue
        lobe_c = lobe.mean(axis=0)
        out.append(_landmark(f"optic lobe {letter}", lobe))
        radius = float(np.percentile(np.linalg.norm(lobe - lobe_c, axis=1), LOBE_RADIUS_PERCENTILE))

        direction = None
        lamina = _xyz(n[on_side & is_lamina])
        if len(lamina):
            out.append(_landmark(f"lamina {letter}", lamina))
            delta = lamina.mean(axis=0) - lobe_c
            norm = float(np.linalg.norm(delta))
            if norm > 0:
                unit = delta / norm
                direction = [float(v) for v in unit]
        sides[side] = {
            "lobe_center": [float(v) for v in lobe_c],
            "radius": radius,
            "lamina_direction": direction,
        }
    return {"landmarks": out, "sides": sides}
