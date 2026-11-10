"""Tool functions: small JSON-friendly outputs only. No MCP imports here."""

from flybrain.atlas import Atlas, clean_filter
from flybrain.selections import SelectionStore

SAMPLE_SIZE = 5
MAX_PAGE = 50
MAX_TOP = 100  # count_neurons never shows more groups than this

# tool filter name -> atlas column, for the closed sets we can list in a no-match hint
CLOSED_FILTERS = {"nt": "top_nt", "super_class": "super_class", "side": "side"}


def _int(name: str, value) -> int:
    """Real ints only: bool is not a count, floats/NaN/strings are rejected loudly."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _clamp(name: str, value: int, low: int, high: int | None, notes: list[str]) -> int:
    """Clamp into [low, high] and say so in `notes` (no silent changes)."""
    if value < low:
        notes.append(f"{name} {value} clamped to {low}")
        return low
    if high is not None and value > high:
        notes.append(f"{name} capped to {high}")
        return high
    return value


def _clean_filters(cell_type, nt, super_class, side) -> dict:
    """Validated, stripped filters with empty ones dropped."""
    given = {"cell_type": cell_type, "nt": nt, "super_class": super_class, "side": side}
    cleaned = {k: clean_filter(k, v) for k, v in given.items()}
    return {k: v for k, v in cleaned.items() if v is not None}


def _no_match_hint(atlas: Atlas, filters: dict) -> str:
    """Explain an empty result; list valid values when a closed-set filter has a bad value."""
    parts = []
    for name, column in CLOSED_FILTERS.items():
        if name in filters:
            valid = atlas.distinct(column)
            if filters[name].lower() not in {v.lower() for v in valid}:
                parts.append(f"{name}={filters[name]!r} is not a known value; valid {name}: {', '.join(valid)}")
    if parts:
        return "No neurons matched. " + "; ".join(parts) + "."
    return (
        "No neurons matched. Drop a filter, use a shorter cell_type prefix "
        "(it is a prefix match), or check count_neurons to see valid values."
    )


def search_neurons(
    atlas: Atlas, store: SelectionStore, cell_type=None, nt=None, super_class=None, side=None
) -> dict:
    """Find neurons, keep ALL matches server-side under a ticket, return a tiny summary."""
    filters = _clean_filters(cell_type, nt, super_class, side)
    hits = atlas.find(**filters)
    total = len(hits)
    if total == 0:
        return {
            "selection_id": None,
            "total": 0,
            "filters": filters,
            "sample": [],
            "by_super_class": {},
            "hint": _no_match_hint(atlas, filters),
        }
    description = ", ".join(f"{k}={v}" for k, v in filters.items()) or "whole brain"
    selection_id = store.add(hits["root_id"], description, filters)
    by_class = hits["super_class"].fillna("unknown").value_counts().head(8)
    return {
        "selection_id": selection_id,
        "total": total,
        "filters": filters,
        "sample": [atlas.describe(int(r)) for r in hits["root_id"].head(SAMPLE_SIZE)],
        "by_super_class": {str(k): int(v) for k, v in by_class.items()},
    }


def count_neurons(
    atlas: Atlas, group_by, cell_type=None, nt=None, super_class=None, side=None, top=20
) -> dict:
    """Counts per value of `group_by`: top N groups plus the sum of the rest."""
    notes: list[str] = []
    top = _clamp("top", _int("top", top), 0, MAX_TOP, notes)
    filters = _clean_filters(cell_type, nt, super_class, side)
    counts = atlas.count_by(group_by, **filters)  # ValueError lists allowed fields
    items = list(counts.items())
    out = {
        "group_by": group_by,
        "total": sum(counts.values()),
        "counts": dict(items[:top]),
        "other": sum(v for _, v in items[top:]),
    }
    if notes:
        out["note"] = "; ".join(notes)
    return out


def describe_neuron(atlas: Atlas, root_id: int) -> dict:
    return atlas.describe(root_id)  # atlas validates the id and words the errors


def list_selection(
    atlas: Atlas, store: SelectionStore, selection_id, offset=0, limit=20
) -> dict:
    """One page of a ticket. Limit is hard-capped so the context stays small."""
    if not isinstance(selection_id, str):
        raise TypeError(f"selection_id must be a string, got {type(selection_id).__name__}")
    notes: list[str] = []
    offset = _clamp("offset", _int("offset", offset), 0, None, notes)
    limit = _clamp("limit", _int("limit", limit), 0, MAX_PAGE, notes)
    sel = store.get(selection_id)
    total = len(sel.root_ids)
    if offset >= total:
        notes.append(f"offset {offset} is past the end (total {total})")
    page = sel.root_ids[offset : offset + limit]
    out = {
        "selection_id": selection_id,
        "total": total,
        "offset": offset,
        "neurons": [atlas.describe(r) for r in page],
    }
    if notes:
        out["note"] = "; ".join(notes)
    return out
