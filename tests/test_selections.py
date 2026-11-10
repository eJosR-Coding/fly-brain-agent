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


# --- hardening (H1): LRU cap with expiry hints ------------------------------------

def test_default_cap_is_100(monkeypatch):
    monkeypatch.delenv("FLYBRAIN_MAX_SELECTIONS", raising=False)
    assert SelectionStore().max_items == 100


def test_cap_comes_from_env(monkeypatch):
    monkeypatch.setenv("FLYBRAIN_MAX_SELECTIONS", "7")
    assert SelectionStore().max_items == 7


@pytest.mark.parametrize("bad", ["0", "-2", "abc", "1.5"])
def test_bad_env_cap_is_a_clear_error(monkeypatch, bad):
    monkeypatch.setenv("FLYBRAIN_MAX_SELECTIONS", bad)
    with pytest.raises(ValueError, match="FLYBRAIN_MAX_SELECTIONS"):
        SelectionStore()


def test_selection_keeps_the_filters_that_made_it():
    store = SelectionStore(max_items=5)
    sid = store.add([1], "nt=dopamine", filters={"nt": "dopamine"})
    assert store.get(sid).filters == {"nt": "dopamine"}


def test_oldest_ticket_is_evicted_over_the_cap():
    store = SelectionStore(max_items=2)
    a = store.add([1], "a", filters={"nt": "gaba"})
    store.add([2], "b")
    store.add([3], "c")
    with pytest.raises(KeyError, match=f"{a}.*expired"):
        store.get(a)


def test_get_refreshes_recency():
    store = SelectionStore(max_items=2)
    a = store.add([1], "a")
    b = store.add([2], "b")
    store.get(a)  # a is now the freshest
    store.add([3], "c")  # evicts b, not a
    assert store.get(a).root_ids == (1,)
    with pytest.raises(KeyError, match="expired"):
        store.get(b)


def test_expired_message_names_the_search_to_rerun():
    store = SelectionStore(max_items=1)
    a = store.add([1], "x", filters={"nt": "dopamine", "side": "left"})
    store.add([2], "y")
    with pytest.raises(KeyError) as exc:
        store.get(a)
    msg = exc.value.args[0]
    assert f"selection '{a}' expired (least recently used)" in msg
    assert "search_neurons(nt='dopamine', side='left')" in msg
    assert "Run that search again." in msg


def test_expired_whole_brain_ticket_says_no_filters():
    store = SelectionStore(max_items=1)
    a = store.add([1], "whole brain", filters={})
    store.add([2], "y")
    with pytest.raises(KeyError, match=r"search_neurons\(\)"):
        store.get(a)


def test_unknown_ticket_lists_at_most_10_recent_ids():
    store = SelectionStore(max_items=100)
    ids = [store.add([i], "x") for i in range(25)]
    with pytest.raises(KeyError) as exc:
        store.get("sel_nope00")
    msg = exc.value.args[0]
    assert "unknown selection 'sel_nope00'" in msg
    assert ids[-1] in msg and ids[-10] in msg
    assert ids[0] not in msg and ids[-11] not in msg
    assert "and 15 more" in msg
