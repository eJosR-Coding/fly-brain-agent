"""The descending-neuron traces the body drive needs, computed on demand from a recording (pure, no flygym import).

A recording stores sparse spike counts per atlas row and time bin (ActivityRecord.frames, summed over trials), so the
six traces the body follows can be computed from ANY existing recording, without re-running it and without changing
what stimulate/fly store: dng100 (both DNg100 neurons), dna01_left / dna01_right and dna02_left / dna02_right (one
neuron of each steering cell type on each side) and dnp01 (both giant-fiber neurons). The groups are looked up in the
atlas by exact cell_type and side. The rate is the same mean Hz per neuron as the recorded readouts:
spikes / (neurons * trials * bin_s).
"""

from __future__ import annotations

import numpy as np

# readout name -> (cell_type, side or None for both sides); names are the ones body_drive reads
DN_GROUPS = {
    "dng100": ("DNg100", None),
    "dna01_left": ("DNa01", "left"),
    "dna01_right": ("DNa01", "right"),
    "dna02_left": ("DNa02", "left"),
    "dna02_right": ("DNa02", "right"),
    "dnp01": ("DNp01", None),
}


def dn_group_rows(atlas) -> dict[str, np.ndarray]:
    """Atlas rows of each DN group (an empty array when the atlas has no such neuron)."""
    n = atlas.neurons
    cell_type = n["cell_type"].to_numpy(dtype=object)
    side = n["side"].to_numpy(dtype=object)
    out = {}
    for name, (ct, sd) in DN_GROUPS.items():
        mask = cell_type == ct
        if sd is not None:
            mask &= side == sd
        out[name] = np.flatnonzero(mask).astype(np.int64)
    return out


def dn_readouts(record, atlas) -> list[dict]:
    """[{"name", "label", "neurons", "rows", "hz"}] for every DN group present in the atlas, from `record`'s frames.

    `record` needs rows (atlas rows of the frames), frames (n_rows, n_bins), bin_ms and trials."""
    rows = np.asarray(record.rows)
    frames = np.asarray(record.frames)
    bin_s = float(record.bin_ms) / 1000.0
    trials = max(1, int(record.trials))
    out = []
    for name, group in dn_group_rows(atlas).items():
        if len(group) == 0:
            continue
        keep = np.isin(rows, group)
        counts = frames[keep].sum(axis=0, dtype=np.float64) if keep.any() else np.zeros(frames.shape[1])
        hz = counts / (len(group) * trials * bin_s)
        out.append({"name": name, "label": name.upper(), "neurons": int(len(group)),
                    "rows": [int(r) for r in group], "hz": [round(float(v), 2) for v in hz]})
    return out
