import json

import pytest

from flybrain import tools
from flybrain.connectome import ConnectomeError
from flybrain.selections import SelectionStore, UnknownSelectionError


@pytest.fixture
def store():
    return SelectionStore()


def _ticket(atlas, store, **filters):
    return tools.search_neurons(atlas, store, **filters)["selection_id"]


def test_neighbors_out_lists_partners_and_makes_a_ticket(tiny_atlas, store, tiny_conn):
    sid = _ticket(tiny_atlas, store, cell_type="EPG")
    out = tools.neighbors(tiny_atlas, store, tiny_conn, sid, direction="out")
    assert out["partners_total"] == 1
    assert out["top"][0] == {"root_id": 1, "cell_type": "PEN_a", "synapses": 10}
    assert store.get(out["selection_id"]).root_ids == (1,)
    assert out["by_super_class"] == {"central": 1}
    assert "structural" in out["note"].lower() and "5" in out["note"]


def test_neighbors_in_direction(tiny_atlas, store, tiny_conn):
    sid = _ticket(tiny_atlas, store, cell_type="EPG")
    out = tools.neighbors(tiny_atlas, store, tiny_conn, sid, direction="in")
    assert [p["root_id"] for p in out["top"]] == [101]


def test_neighbors_clamps_top_and_says_so(tiny_atlas, store, tiny_conn):
    sid = _ticket(tiny_atlas, store, cell_type="EPG")
    out = tools.neighbors(tiny_atlas, store, tiny_conn, sid, top=1000)
    assert "capped to 50" in out["note"]


def test_neighbors_without_partners_has_no_ticket_and_a_hint(tiny_atlas, store, tiny_conn):
    sid = _ticket(tiny_atlas, store, cell_type="ORN")
    out = tools.neighbors(tiny_atlas, store, tiny_conn, sid, direction="in")
    assert out["selection_id"] is None and out["partners_total"] == 0 and out["hint"]


def test_neighbors_bad_direction_is_a_guiding_error(tiny_atlas, store, tiny_conn):
    sid = _ticket(tiny_atlas, store, cell_type="EPG")
    with pytest.raises(ValueError, match="direction"):
        tools.neighbors(tiny_atlas, store, tiny_conn, sid, direction="up")


def test_tools_without_a_connectome_say_how_to_build_it(tiny_atlas, store):
    sid = _ticket(tiny_atlas, store, cell_type="EPG")
    with pytest.raises(ConnectomeError, match="flybrain-build-connectome"):
        tools.neighbors(tiny_atlas, store, None, sid)
    with pytest.raises(ConnectomeError, match="flybrain-build-connectome"):
        tools.top_hubs(tiny_atlas, store, None)


def test_unknown_ticket_is_the_usual_guiding_error(tiny_atlas, store, tiny_conn):
    with pytest.raises(UnknownSelectionError):
        tools.neighbors(tiny_atlas, store, tiny_conn, "sel_nope00")


def test_find_path_returns_hops_path_and_ticket(tiny_atlas, store, tiny_conn):
    a = _ticket(tiny_atlas, store, cell_type="ORN_DA1")
    b = tools.search_neurons(tiny_atlas, store, cell_type="PEN")["selection_id"]
    out = tools.find_path(tiny_atlas, store, tiny_conn, a, b)
    assert out["found"] and out["hops"] == 3
    assert [p["root_id"] for p in out["path"]] == [102, 101, 100, 1]
    assert out["path"][0]["cell_type"] == "ORN_DA1"
    assert store.get(out["selection_id"]).root_ids == (102, 101, 100, 1)
    assert "structural" in out["note"].lower()


def test_find_path_not_found_gives_hint_and_no_ticket(tiny_atlas, store, tiny_conn):
    a = _ticket(tiny_atlas, store, cell_type="EPG")
    b = _ticket(tiny_atlas, store, cell_type="ORN")
    out = tools.find_path(tiny_atlas, store, tiny_conn, a, b)
    assert out["found"] is False and out["selection_id"] is None and out["hint"]


def test_find_path_clamps_max_hops(tiny_atlas, store, tiny_conn):
    a = _ticket(tiny_atlas, store, cell_type="EPG")
    out = tools.find_path(tiny_atlas, store, tiny_conn, a, a, max_hops=99)
    assert "max_hops capped to 12" in out["note"]


def test_lesion_effect_counts_lost_targets_and_tickets_them(tiny_atlas, store, tiny_conn):
    src = _ticket(tiny_atlas, store, cell_type="EPG")
    tgt = tools.search_neurons(tiny_atlas, store, cell_type="PEN_a")["selection_id"]
    # removing the whole PEN_a class kills every route through rows 0, 1, 2, 3
    out = tools.lesion_effect(tiny_atlas, store, tiny_conn, src, tgt, tgt)
    assert out["targets_reachable_before"] == 4 and out["targets_reachable_after"] == 0
    assert out["lost"] == 4 and "lost_share" in out
    assert len(store.get(out["selection_id"]).root_ids) == 4
    assert "structural" in out["note"].lower()


def test_lesion_effect_with_nothing_lost_has_no_ticket(tiny_atlas, store, tiny_conn):
    src = _ticket(tiny_atlas, store, cell_type="EPG")
    removed = _ticket(tiny_atlas, store, cell_type="ORN")
    out = tools.lesion_effect(tiny_atlas, store, tiny_conn, src, src, removed)
    assert out["lost"] == 0 and out["selection_id"] is None


def test_top_hubs_ranks_by_synapse_strength(tiny_atlas, store, tiny_conn):
    out = tools.top_hubs(tiny_atlas, store, tiny_conn, metric="total", top=2)
    assert [h["root_id"] for h in out["hubs"]] == [2, 1]  # row 1 (28) and row 0 (20)
    assert out["hubs"][0]["synapses"] == 28
    assert store.get(out["selection_id"]).root_ids == (2, 1)


def test_top_hubs_filters_candidates(tiny_atlas, store, tiny_conn):
    out = tools.top_hubs(tiny_atlas, store, tiny_conn, metric="out", cell_type="EPG")
    assert [h["root_id"] for h in out["hubs"]] == [100] and out["filters"] == {"cell_type": "EPG"}


def test_top_hubs_bad_metric_lists_the_valid_ones(tiny_atlas, store, tiny_conn):
    with pytest.raises(ValueError, match="in, out, total"):
        tools.top_hubs(tiny_atlas, store, tiny_conn, metric="betweenness")


def test_default_outputs_are_small_plain_json(tiny_atlas, store, tiny_conn):
    sid = _ticket(tiny_atlas, store, cell_type="PEN")
    for out in (
        tools.neighbors(tiny_atlas, store, tiny_conn, sid, direction="both"),
        tools.find_path(tiny_atlas, store, tiny_conn, sid, sid),
        tools.lesion_effect(tiny_atlas, store, tiny_conn, sid, sid, sid),
        tools.top_hubs(tiny_atlas, store, tiny_conn),
    ):
        assert len(json.dumps(out)) < 2000


def test_derived_tickets_say_where_they_came_from_when_they_expire(tiny_atlas, tiny_conn):
    small = SelectionStore(max_items=1)
    sid = _ticket(tiny_atlas, small, cell_type="EPG")
    out = tools.neighbors(tiny_atlas, small, tiny_conn, sid)  # evicts sid, keeps the partners ticket
    tools.search_neurons(tiny_atlas, small, cell_type="EPG")  # evicts the partners ticket
    with pytest.raises(UnknownSelectionError, match=r"neighbors\(.*\)"):
        small.get(out["selection_id"])


def test_no_partner_hint_does_not_suggest_the_direction_already_used(tiny_atlas, store, tiny_conn):
    sid = store.add([31], "an unconnected neuron")  # root_id 31 -> row 30, not in the graph
    out = tools.neighbors(tiny_atlas, store, tiny_conn, sid, direction="both")
    assert out["partners_total"] == 0 and "direction='both'" not in out["hint"]
    assert "direction='both'" in tools.neighbors(tiny_atlas, store, tiny_conn, sid, direction="out")["hint"]
