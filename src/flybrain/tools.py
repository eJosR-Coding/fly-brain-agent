"""Tool functions: small JSON-friendly outputs only. No MCP imports here."""

import numpy as np

from flybrain.atlas import Atlas, RowIndex, clean_filter
from flybrain.connectome import Connectome, ConnectomeError
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


# --- connectome tools (structural graph queries) ---------------------------------------

MAX_PARTNERS = 50  # neighbors / top_hubs never list more than this
MAX_HOPS = 12
SUMMARY_GROUPS = 6
STRUCTURAL = (
    "Structural connections only (>= {n} synapses between a neuron pair, summed over "
    "neuropils); this is wiring, not signal flow, sign or function."
)
_ROW_INDEX_CACHE: dict[int, tuple[Atlas, RowIndex]] = {}


def _need_connectome(conn) -> Connectome:
    if conn is None:
        raise ConnectomeError(
            "connectome not available: run flybrain-build-connectome first "
            "(set FLYBRAIN_CONNECTOME to the built .npz if it is not data/processed/connectome_783.npz)"
        )
    return conn


def _row_index(atlas: Atlas) -> RowIndex:
    """One RowIndex per atlas, built on first use (hashing 139k ids is not free)."""
    hit = _ROW_INDEX_CACHE.get(id(atlas))
    if hit is None or hit[0] is not atlas:
        if len(_ROW_INDEX_CACHE) >= 4:
            _ROW_INDEX_CACHE.clear()
        hit = _ROW_INDEX_CACHE[id(atlas)] = (atlas, RowIndex(atlas))
    return hit[1]


def _selection_rows(atlas: Atlas, store: SelectionStore, selection_id) -> np.ndarray:
    if not isinstance(selection_id, str):
        raise TypeError(f"selection_id must be a string, got {type(selection_id).__name__}")
    sel = store.get(selection_id)
    rows = _row_index(atlas).positions(sel.root_ids)
    rows = rows[rows >= 0]
    if not len(rows):
        raise ValueError(f"selection {selection_id!r} has no neurons in this atlas; run search_neurons again")
    return rows


def _root_ids(atlas: Atlas, rows) -> list[int]:
    return [int(r) for r in atlas.neurons["root_id"].to_numpy()[np.asarray(rows, dtype=np.int64)]]


def _label(atlas: Atlas, rows, extra: dict | None = None) -> list[dict]:
    """[{root_id, cell_type, ...extra}] for rows, in order (cell_type None when unannotated)."""
    n = atlas.neurons
    out = []
    for i, r in enumerate(rows):
        item = {"root_id": int(n["root_id"].iat[int(r)]), "cell_type": _text(n["cell_type"].iat[int(r)])}
        for key, values in (extra or {}).items():
            item[key] = int(values[i])
        out.append(item)
    return out


def _text(value):
    return None if value is None or value != value else str(value)


def _summaries(atlas: Atlas, rows) -> dict:
    sub = atlas.neurons.iloc[np.asarray(rows, dtype=np.int64)]
    return {
        "by_super_class": {
            str(k): int(v) for k, v in sub["super_class"].fillna("unknown").value_counts().head(SUMMARY_GROUPS).items()
        },
        "by_cell_type": {
            str(k): int(v) for k, v in sub["cell_type"].fillna("unknown").value_counts().head(SUMMARY_GROUPS).items()
        },
    }


def _ticket(atlas: Atlas, store: SelectionStore, rows, description: str, source: str) -> str:
    return store.add(_root_ids(atlas, rows), description, {"_source": source})


def _finish(out: dict, notes: list[str], conn: Connectome) -> dict:
    notes = [*notes, STRUCTURAL.format(n=conn.min_synapses)]
    out["note"] = "; ".join(notes)
    return out


def neighbors(
    atlas: Atlas, store: SelectionStore, conn, selection_id, direction="out", top=10
) -> dict:
    """Partners of a selection (out = it projects to them, in = they project onto it)."""
    conn = _need_connectome(conn)
    notes: list[str] = []
    top = _clamp("top", _int("top", top), 1, MAX_PARTNERS, notes)
    rows = _selection_rows(atlas, store, selection_id)
    partner_rows, syn = conn.neighbors(rows, direction)  # ValueError lists directions
    out = {
        "selection_id": None,
        "direction": direction,
        "input_neurons": int(len(rows)),
        "partners_total": int(len(partner_rows)),
        "synapses_total": int(syn.sum()),
    }
    if not len(partner_rows):
        retry = " Try direction='both' or another selection." if direction != "both" else " Try another selection."
        out["hint"] = f"No {direction} partners at >= {conn.min_synapses} synapses.{retry}"
        out["top"] = []
        return _finish(out, notes, conn)
    out["selection_id"] = _ticket(
        atlas, store, partner_rows, f"{direction}-partners of {selection_id}",
        f"neighbors(selection_id={selection_id!r}, direction={direction!r})",
    )
    out["top"] = _label(atlas, partner_rows[:top], {"synapses": syn[:top]})
    out |= _summaries(atlas, partner_rows)
    return _finish(out, notes, conn)


def find_path(
    atlas: Atlas, store: SelectionStore, conn, from_selection, to_selection, max_hops=8
) -> dict:
    """One shortest directed structural path from any neuron of one selection to any of another."""
    conn = _need_connectome(conn)
    notes: list[str] = []
    max_hops = _clamp("max_hops", _int("max_hops", max_hops), 1, MAX_HOPS, notes)
    src = _selection_rows(atlas, store, from_selection)
    tgt = _selection_rows(atlas, store, to_selection)
    found = conn.shortest_path(src, tgt, max_hops)
    out = {"found": found is not None, "from_neurons": int(len(src)), "to_neurons": int(len(tgt)),
           "max_hops": max_hops, "selection_id": None}
    if found is None:
        out["hint"] = (
            f"No directed structural path within {max_hops} hops. Raise max_hops (up to {MAX_HOPS}), "
            "swap from/to (paths are directed), or widen the groups."
        )
        return _finish(out, notes, conn)
    hops, path = found
    out["hops"] = hops
    labelled = _label(atlas, path)
    n = atlas.neurons
    for item, r in zip(labelled, path):
        item["super_class"] = _text(n["super_class"].iat[r])
    out["path"] = labelled
    out["selection_id"] = _ticket(
        atlas, store, path, f"path {from_selection} -> {to_selection}",
        f"find_path(from_selection={from_selection!r}, to_selection={to_selection!r}, max_hops={max_hops})",
    )
    return _finish(out, notes, conn)


def lesion_effect(
    atlas: Atlas, store: SelectionStore, conn, source_selection, target_selection,
    removed_selection, max_hops=8,
) -> dict:
    """Virtual lesion: how many targets stay structurally reachable from the sources without `removed`."""
    conn = _need_connectome(conn)
    notes: list[str] = []
    max_hops = _clamp("max_hops", _int("max_hops", max_hops), 1, MAX_HOPS, notes)
    src = _selection_rows(atlas, store, source_selection)
    tgt = _selection_rows(atlas, store, target_selection)
    rem = _selection_rows(atlas, store, removed_selection)
    res = conn.lesion_effect(src, tgt, rem, max_hops)
    before = res["targets_reachable_before"]
    out = {
        "source_neurons": int(len(src)),
        "target_neurons": res["targets_total"],
        "removed_neurons": int(len(rem)),
        "max_hops": max_hops,
        "targets_reachable_before": before,
        "targets_reachable_after": res["targets_reachable_after"],
        "lost": res["lost"],
        "lost_share": round(res["lost"] / before, 3) if before else 0.0,
        "selection_id": None,
    }
    if res["lost"]:
        lost = res["lost_rows"]
        out["selection_id"] = _ticket(
            atlas, store, lost, f"targets lost after removing {removed_selection}",
            f"lesion_effect(source_selection={source_selection!r}, target_selection={target_selection!r}, "
            f"removed_selection={removed_selection!r}, max_hops={max_hops})",
        )
        out["sample_lost"] = _label(atlas, lost[:5])
        out["lost_by_cell_type"] = _summaries(atlas, lost)["by_cell_type"]
    elif before == 0:
        out["hint"] = (
            f"None of the targets was reachable from the sources within {max_hops} hops even before "
            "the lesion; check with find_path first."
        )
    return _finish(out, notes, conn)


def top_hubs(
    atlas: Atlas, store: SelectionStore, conn, metric="total", top=10,
    cell_type=None, nt=None, super_class=None, side=None,
) -> dict:
    """Neurons with the highest summed synapse count (in, out or total), optionally filtered."""
    conn = _need_connectome(conn)
    notes: list[str] = []
    top = _clamp("top", _int("top", top), 1, MAX_PARTNERS, notes)
    filters = _clean_filters(cell_type, nt, super_class, side)
    candidates = None
    if filters:
        hits = atlas.find(**filters)
        if hits.empty:
            return _finish({"selection_id": None, "metric": metric, "filters": filters, "hubs": [],
                            "hint": _no_match_hint(atlas, filters)}, notes, conn)
        candidates = hits.index.to_numpy()
    rows, strength = conn.hubs(metric, top, candidates)  # ValueError lists metrics
    out = {
        "metric": metric,
        "filters": filters,
        "hubs": _label(atlas, rows, {"synapses": strength}),
        "selection_id": _ticket(
            atlas, store, rows, f"top {len(rows)} hubs by {metric} synapses",
            f"top_hubs(metric={metric!r}, top={top}, filters={filters})",
        ),
    }
    super_classes = atlas.neurons["super_class"].to_numpy()
    for item, r in zip(out["hubs"], rows):
        item["super_class"] = _text(super_classes[int(r)])
    return _finish(out, notes, conn)
