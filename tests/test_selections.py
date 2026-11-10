import re

import pytest

from flybrain.selections import SelectionStore


def test_ids_are_short_unique_tickets():
    store = SelectionStore()
    ids = [store.add([i], f"group {i}") for i in range(200)]
    assert len(set(ids)) == 200
    assert all(re.fullmatch(r"sel_[0-9a-f]{6}", sid) for sid in ids)


def test_a_restarted_store_never_reuses_an_old_ticket():
    # restart = brand new store; the old ticket must fail loudly, not point at other neurons
    before = SelectionStore()
    old_id = before.add([1, 2, 3], "dopamine neurons")
    after = SelectionStore()
    new_id = after.add([9, 9, 9], "optic neurons")
    assert new_id != old_id
    with pytest.raises(KeyError, match="unknown selection"):
        after.get(old_id)


def test_get_returns_what_was_stored():
    store = SelectionStore()
    sid = store.add([5, 6, 7], "some neurons")
    sel = store.get(sid)
    assert sel.id == sid and sel.description == "some neurons"
    assert list(sel.root_ids) == [5, 6, 7]


def test_unknown_ticket_raises_helpful_key_error():
    store = SelectionStore()
    known = store.add([1], "x")
    with pytest.raises(KeyError, match=f"sel_99.*{known}"):
        store.get("sel_99")
