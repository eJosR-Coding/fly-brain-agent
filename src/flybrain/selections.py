"""Selection tickets: big result sets live server-side, the LLM only holds an id."""

import secrets
from dataclasses import dataclass


@dataclass(frozen=True)
class Selection:
    id: str
    description: str
    root_ids: tuple[int, ...]


class SelectionStore:
    """In-memory, session-scoped store of selections.

    Ids are random (sel_3f9a2c), never a counter: a counter restarts at sel_1 after a server
    restart, so a stale ticket would silently point at different neurons. Random ids make a
    stale ticket fail loudly with KeyError instead. No cap on the vibes, just on collisions.
    """

    def __init__(self) -> None:
        self._items: dict[str, Selection] = {}

    def add(self, root_ids, description: str) -> str:
        selection_id = self._new_id()
        self._items[selection_id] = Selection(
            selection_id, description, tuple(int(r) for r in root_ids)
        )
        return selection_id

    def _new_id(self) -> str:
        while True:  # 16.7M possible ids; a clash inside one session is basically never, but check anyway
            candidate = f"sel_{secrets.token_hex(3)}"
            if candidate not in self._items:
                return candidate

    def get(self, selection_id: str) -> Selection:
        try:
            return self._items[selection_id]
        except KeyError:
            known = ", ".join(self._items) or "none yet"
            raise KeyError(
                f"unknown selection {selection_id!r}; known selections: {known}. "
                "Run search_neurons to get a fresh selection_id."
            ) from None
