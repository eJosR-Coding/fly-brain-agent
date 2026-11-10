"""Atlas core: where every FlyWire neuron is and what it is. Pandas/numpy only."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VOXEL_NM = (4, 4, 40)  # FlyWire voxel size in nm (x, y, z)

TEXT_FIELDS = ("super_class", "cell_class", "cell_type", "top_nt", "side", "flow")
COLUMNS = ["root_id", "x_um", "y_um", "z_um", *TEXT_FIELDS]


class AtlasLoadError(ValueError):
    """The annotations file cannot be turned into an atlas."""


class UnknownNeuronError(KeyError):
    """No neuron with that root_id. str() gives the plain message, not a quoted repr."""

    def __str__(self) -> str:
        return str(self.args[0])


MAX_FILTER_CHARS = 200  # longer filter strings are rejected (keeps echoes and errors small)


def load_atlas(path) -> "Atlas":
    """Read the annotations TSV and convert voxel coords to micrometers.

    Every way the file can be wrong ends in ONE error type, AtlasLoadError, with a
    one-line message (no pandas traceback leaking to the user).
    """
    if path is None:
        raise AtlasLoadError("no annotations path given (set FLYBRAIN_ANNOTATIONS)")
    path = Path(path)
    if path.is_dir():
        raise AtlasLoadError(f"annotations path is a directory, not a file: {path}")
    if not path.exists():
        raise AtlasLoadError(f"annotations file not found: {path}")
    if path.stat().st_size == 0:
        raise AtlasLoadError(f"annotations file is empty: {path}")

    needed = ["root_id", "pos_x", "pos_y", "pos_z", *TEXT_FIELDS]
    try:
        header = pd.read_csv(path, sep="\t", nrows=0).columns
        missing = [c for c in needed if c not in header]
        if missing:
            raise AtlasLoadError(f"missing required columns: {', '.join(missing)}")
        raw = pd.read_csv(path, sep="\t", usecols=needed, dtype=str, low_memory=False)
    except AtlasLoadError:
        raise
    except pd.errors.EmptyDataError:
        raise AtlasLoadError(f"annotations file is empty: {path}") from None
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise AtlasLoadError(f"could not read annotations {path}: {exc}") from None
    if raw.empty:
        raise AtlasLoadError(f"annotations file has no neurons: {path}")

    try:
        root_ids = raw["root_id"].astype("int64")
    except (ValueError, TypeError, OverflowError) as exc:
        raise AtlasLoadError(f"bad root_id (need integers that fit int64): {exc}") from None
    if (root_ids < 0).any():
        raise AtlasLoadError("bad root_id: negative values")
    if root_ids.duplicated().any():
        raise AtlasLoadError("duplicate root_id values in annotations")

    df = pd.DataFrame({"root_id": root_ids})
    for axis, nm in zip("xyz", VOXEL_NM):
        try:
            col = raw[f"pos_{axis}"].astype("float64")
        except (ValueError, TypeError) as exc:
            raise AtlasLoadError(f"non-numeric positions in pos_{axis}: {exc}") from None
        if not np.isfinite(col).all():
            raise AtlasLoadError(f"non-finite or missing positions in pos_{axis}")
        df[f"{axis}_um"] = col * nm / 1000
    for field in TEXT_FIELDS:
        df[field] = raw[field].astype("object").where(raw[field].notna(), None)
    return Atlas(df[COLUMNS].reset_index(drop=True))


def clean_filter(name: str, value) -> str | None:
    """Normalize one filter value: None/''/whitespace -> None (no filter), else stripped str."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    value = value.strip()
    if len(value) > MAX_FILTER_CHARS:
        raise ValueError(f"{name} is too long ({len(value)} chars; max {MAX_FILTER_CHARS})")
    return value or None


def _clean_root_id(root_id) -> int:
    """Accept real ints and all-digit strings; floats lose precision, bools are not ids."""
    if isinstance(root_id, bool) or not isinstance(root_id, (int, str)):
        raise TypeError(f"root_id must be an int, got {type(root_id).__name__}")
    if isinstance(root_id, str):
        if not root_id.isascii() or not root_id.isdigit():
            raise ValueError(f"root_id must be all digits, got {root_id!r}")
        root_id = int(root_id)
    return root_id


@dataclass(frozen=True)
class Atlas:
    neurons: pd.DataFrame

    def _mask(self, cell_type=None, nt=None, super_class=None, side=None) -> pd.Series:
        df = self.neurons
        cell_type = clean_filter("cell_type", cell_type)
        nt = clean_filter("nt", nt)
        super_class = clean_filter("super_class", super_class)
        side = clean_filter("side", side)
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
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError(f"limit must be an int, got {type(limit).__name__}")
            if limit < 0:
                raise ValueError(f"limit must be >= 0, got {limit}")
        hits = self.neurons[self._mask(cell_type, nt, super_class, side)]
        return hits.head(limit) if limit is not None else hits

    def distinct(self, field) -> list[str]:
        """Sorted distinct non-missing values of a text field (valid filter values)."""
        if field not in TEXT_FIELDS:
            raise ValueError(f"unknown field {field!r}; allowed: {', '.join(TEXT_FIELDS)}")
        return sorted(str(v) for v in self.neurons[field].dropna().unique())

    def count_by(self, field, **filters) -> dict[str, int]:
        """Count matching neurons per value of `field`, biggest group first."""
        if not isinstance(field, str) or field not in TEXT_FIELDS:
            raise ValueError(f"unknown field {field!r}; allowed: {', '.join(TEXT_FIELDS)}")
        values = self.find(**filters)[field].fillna("unknown")
        counts = values.value_counts()  # sorted descending
        return {str(k): int(v) for k, v in counts.items()}

    def describe(self, root_id: int) -> dict:
        """One neuron as a plain JSON-friendly dict."""
        root_id = _clean_root_id(root_id)
        rows = self.neurons[self.neurons["root_id"] == root_id]
        if rows.empty:
            raise UnknownNeuronError(
                f"no neuron with root_id {root_id}; get ids from search_neurons or list_selection"
            )
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


class RowIndex:
    """root_id -> atlas row, built once (the browser only knows rows, never root_ids).

    Unknown ids are ignored and counted: a stale or foreign id must not break a highlight.
    """

    def __init__(self, atlas: Atlas) -> None:
        self._index = pd.Index(atlas.neurons["root_id"].to_numpy())

    def lookup(self, root_ids) -> tuple[np.ndarray, int]:
        """Return (row indices as little-endian uint32, how many ids were not found)."""
        found = self._index.get_indexer(np.asarray(list(root_ids), dtype="int64"))
        hit = found[found >= 0]
        return hit.astype("<u4"), int((found < 0).sum())

    def positions(self, root_ids) -> np.ndarray:
        """Atlas row per id, -1 where unknown (aligned with the input, unlike lookup)."""
        return self._index.get_indexer(np.asarray(root_ids, dtype="int64"))

    def __len__(self) -> int:
        return len(self._index)
