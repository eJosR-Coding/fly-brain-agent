"""Tool functions: small JSON-friendly outputs only. No MCP imports here."""

from flybrain.atlas import Atlas
from flybrain.selections import SelectionStore

SAMPLE_SIZE = 5
MAX_PAGE = 50


def search_neurons(
    atlas: Atlas, store: SelectionStore, cell_type=None, nt=None, super_class=None, side=None
) -> dict:
    """Find neurons, keep ALL matches server-side under a ticket, return a tiny summary."""
    given = {"cell_type": cell_type, "nt": nt, "super_class": super_class, "side": side}
    filters = {k: v for k, v in given.items() if v is not None}
    hits = atlas.find(**filters)
    total = len(hits)
    if total == 0:
        return {
            "selection_id": None,
            "total": 0,
            "filters": filters,
            "sample": [],
            "by_super_class": {},
            "hint": "No neurons matched. Drop a filter, use a shorter cell_type prefix "
            "(it is a prefix match), or check count_neurons to see valid values.",
        }
    description = ", ".join(f"{k}={v}" for k, v in filters.items()) or "whole brain"
    selection_id = store.add(hits["root_id"], description)
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
    given = {"cell_type": cell_type, "nt": nt, "super_class": super_class, "side": side}
    filters = {k: v for k, v in given.items() if v is not None}
    counts = atlas.count_by(group_by, **filters)  # ValueError lists allowed fields
    items = list(counts.items())
    shown = dict(items[: max(top, 0)])
    return {
        "group_by": group_by,
        "total": sum(counts.values()),
        "counts": shown,
        "other": sum(v for _, v in items[max(top, 0):]),
    }


def describe_neuron(atlas: Atlas, root_id: int) -> dict:
    return atlas.describe(int(root_id))


def list_selection(
    atlas: Atlas, store: SelectionStore, selection_id, offset=0, limit=20
) -> dict:
    """One page of a ticket. Limit is hard-capped so the context stays small."""
    sel = store.get(selection_id)
    offset = max(int(offset), 0)
    limit = min(max(int(limit), 0), MAX_PAGE)
    page = sel.root_ids[offset : offset + limit]
    return {
        "selection_id": selection_id,
        "total": len(sel.root_ids),
        "offset": offset,
        "neurons": [atlas.describe(r) for r in page],
    }
