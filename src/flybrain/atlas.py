"""Atlas core: where every FlyWire neuron is and what it is. Pandas/numpy only."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

VOXEL_NM = (4, 4, 40)  # FlyWire voxel size in nm (x, y, z)

TEXT_FIELDS = ("super_class", "cell_class", "cell_type", "top_nt", "side", "flow")
COLUMNS = ["root_id", "x_um", "y_um", "z_um", *TEXT_FIELDS]


def load_atlas(path) -> "Atlas":
    """Read the annotations TSV and convert voxel coords to micrometers."""
    usecols = ["root_id", "pos_x", "pos_y", "pos_z", *TEXT_FIELDS]
    raw = pd.read_csv(
        Path(path), sep="\t", usecols=usecols, dtype={"root_id": "int64"}, low_memory=False
    )
    df = pd.DataFrame({"root_id": raw["root_id"]})
    for axis, nm in zip("xyz", VOXEL_NM):
        df[f"{axis}_um"] = raw[f"pos_{axis}"].astype("float64") * nm / 1000
    for field in TEXT_FIELDS:
        df[field] = raw[field].astype("object").where(raw[field].notna(), None)
    return Atlas(df[COLUMNS].reset_index(drop=True))


@dataclass(frozen=True)
class Atlas:
    neurons: pd.DataFrame

    def _mask(self, cell_type=None, nt=None, super_class=None, side=None) -> pd.Series:
        df = self.neurons
        mask = pd.Series(True, index=df.index)
        if cell_type is not None:
            starts = df["cell_type"].str.lower().str.startswith(cell_type.lower())
            mask &= starts.fillna(False).astype(bool)
        for column, wanted in (("top_nt", nt), ("super_class", super_class), ("side", side)):
            if wanted is not None:
                same = df[column].str.lower() == wanted.lower()
                mask &= same.fillna(False).astype(bool)
        return mask

    def find(self, cell_type=None, nt=None, super_class=None, side=None, limit=None) -> pd.DataFrame:
        """Neurons matching ALL given filters (cell_type is a prefix match)."""
        hits = self.neurons[self._mask(cell_type, nt, super_class, side)]
        return hits.head(limit) if limit is not None else hits

    def count_by(self, field, **filters) -> dict[str, int]:
        """Count matching neurons per value of `field`, biggest group first."""
        if field not in TEXT_FIELDS:
            raise ValueError(f"unknown field {field!r}; allowed: {', '.join(TEXT_FIELDS)}")
        values = self.find(**filters)[field].fillna("unknown")
        counts = values.value_counts()  # sorted descending
        return {str(k): int(v) for k, v in counts.items()}

    def describe(self, root_id: int) -> dict:
        """One neuron as a plain JSON-friendly dict."""
        rows = self.neurons[self.neurons["root_id"] == root_id]
        if rows.empty:
            raise KeyError(root_id)
        row = rows.iloc[0]
        out = {"root_id": int(row["root_id"])}
        for axis in ("x_um", "y_um", "z_um"):
            out[axis] = float(row[axis])
        for field in TEXT_FIELDS:
            out[field] = None if pd.isna(row[field]) else str(row[field])
        return out

    def extent_um(self) -> tuple[float, float, float]:
        """Brain size per axis (max - min) in micrometers."""
        n = self.neurons
        return tuple(float(n[c].max() - n[c].min()) for c in ("x_um", "y_um", "z_um"))
