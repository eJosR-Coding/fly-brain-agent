import pytest

from flybrain.selections import SelectionStore


def test_ids_are_sequential_tickets():
    store = SelectionStore()
    assert store.add([1, 2], "first") == "sel_1"
    assert store.add([3], "second") == "sel_2"


def test_get_returns_what_was_stored():
    store = SelectionStore()
    sid = store.add([5, 6, 7], "some neurons")
    sel = store.get(sid)
    assert sel.id == sid and sel.description == "some neurons"
    assert list(sel.root_ids) == [5, 6, 7]


def test_unknown_ticket_raises_helpful_key_error():
    store = SelectionStore()
    store.add([1], "x")
    with pytest.raises(KeyError, match="sel_99.*sel_1"):
        store.get("sel_99")
