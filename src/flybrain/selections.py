"""Selection tickets: big result sets live server-side, the LLM only holds an id."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Selection:
    id: str
    description: str
    root_ids: tuple[int, ...]


class SelectionStore:
    """In-memory store of selections, ids are sel_1, sel_2, ..."""

    def __init__(self) -> None:
        self._items: dict[str, Selection] = {}

    def add(self, root_ids, description: str) -> str:
        selection_id = f"sel_{len(self._items) + 1}"
        self._items[selection_id] = Selection(
            selection_id, description, tuple(int(r) for r in root_ids)
        )
        return selection_id

    def get(self, selection_id: str) -> Selection:
        try:
            return self._items[selection_id]
        except KeyError:
            known = ", ".join(self._items) or "none yet"
            raise KeyError(
                f"unknown selection {selection_id!r}; known selections: {known}. "
                "Run search_neurons to get a fresh selection_id."
            ) from None
