"""Selection tickets: big result sets live server-side, the LLM only holds an id."""

import os
import secrets
from collections import OrderedDict
from dataclasses import dataclass, field

DEFAULT_MAX_SELECTIONS = 100
MAX_TOMBSTONES = 1000  # remembered expired ids (tiny: id -> filters), so we can explain expiry
MAX_KNOWN_SHOWN = 10  # unknown-ticket errors list at most this many recent ids


@dataclass(frozen=True)
class Selection:
    id: str
    description: str
    root_ids: tuple[int, ...]
    filters: dict = field(default_factory=dict)  # what search produced it (for expiry hints)


class UnknownSelectionError(KeyError):
    """Bad or expired ticket. str() is the plain message, not a quoted repr."""

    def __str__(self) -> str:
        return str(self.args[0])


def _max_from_env() -> int:
    raw = os.environ.get("FLYBRAIN_MAX_SELECTIONS")
    if raw is None:
        return DEFAULT_MAX_SELECTIONS
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"FLYBRAIN_MAX_SELECTIONS must be a positive integer, got {raw!r}")
    return value


class SelectionStore:
    """In-memory, session-scoped store of selections.

    Ids are random (sel_3f9a2c), never a counter: a counter restarts at sel_1 after a server
    restart, so a stale ticket would silently point at different neurons. Random ids make a
    stale ticket fail loudly with KeyError instead.

    Capped at ``max_items`` (default 100, env FLYBRAIN_MAX_SELECTIONS): a whole-brain ticket is
    ~2 MB, so we evict the least recently used one. Both add and get refresh recency. An
    evicted ticket fails with a message naming the search that made it, so the model can re-run it.
    """

    def __init__(self, max_items: int | None = None) -> None:
        self.max_items = _max_from_env() if max_items is None else max_items
        if self.max_items < 1:
            raise ValueError(f"max_items must be >= 1, got {self.max_items}")
        self._items: OrderedDict[str, Selection] = OrderedDict()  # oldest first
        self._expired: OrderedDict[str, dict] = OrderedDict()  # evicted id -> filters

    def add(self, root_ids, description: str, filters: dict | None = None) -> str:
        selection_id = self._new_id()
        self._items[selection_id] = Selection(
            selection_id, description, tuple(int(r) for r in root_ids), dict(filters or {})
        )
        while len(self._items) > self.max_items:
            old_id, old = self._items.popitem(last=False)
            self._expired[old_id] = old.filters
            while len(self._expired) > MAX_TOMBSTONES:
                self._expired.popitem(last=False)
        return selection_id

    def _new_id(self) -> str:
        while True:  # 16.7M possible ids; a clash inside one session is basically never, but check anyway
            candidate = f"sel_{secrets.token_hex(3)}"
            if candidate not in self._items:
                return candidate

    def get(self, selection_id: str) -> Selection:
        if selection_id in self._items:
            self._items.move_to_end(selection_id)  # refresh recency
            return self._items[selection_id]
        if selection_id in self._expired:
            filters = self._expired[selection_id]
            if "_source" in filters:  # derived ticket (neighbors, find_path, ...)
                raise UnknownSelectionError(
                    f"selection {selection_id!r} expired (least recently used); it came from "
                    f"{filters['_source']}. Run that call again."
                )
            args = ", ".join(f"{k}={v!r}" for k, v in filters.items())
            raise UnknownSelectionError(
                f"selection {selection_id!r} expired (least recently used); it came from "
                f"search_neurons({args}). Run that search again."
            )
        recent = list(self._items)[-MAX_KNOWN_SHOWN:]
        extra = len(self._items) - len(recent)
        known = ", ".join(recent) or "none yet"
        if extra > 0:
            known += f" and {extra} more"
        raise UnknownSelectionError(
            f"unknown selection {selection_id!r}; known selections: {known}. "
            "Run search_neurons to get a fresh selection_id."
        )
