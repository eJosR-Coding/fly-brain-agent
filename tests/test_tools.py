import json

import pytest

from flybrain import tools
from flybrain.selections import SelectionStore


@pytest.fixture
def store():
    return SelectionStore()


def test_a_huge_search_returns_a_ticket_not_a_list(tiny_atlas, store):
    out = tools.search_neurons(tiny_atlas, store, nt="acetylcholine")
    assert out["total"] == 61
    assert out["selection_id"].startswith("sel_")
    assert len(out["sample"]) == 5
    assert len(store.get(out["selection_id"]).root_ids) == 61
    assert out["filters"] == {"nt": "acetylcholine"}
    assert out["by_super_class"]["central"] == 60


def test_an_empty_search_gives_a_hint_and_no_ticket(tiny_atlas, store):
    out = tools.search_neurons(tiny_atlas, store, cell_type="nope")
    assert out["total"] == 0 and out["selection_id"] is None
    assert out["hint"]


def test_searching_with_no_filters_is_a_whole_brain_ticket(tiny_atlas, store):
    out = tools.search_neurons(tiny_atlas, store)
    assert out["total"] == 63 and out["selection_id"] and out["filters"] == {}


def test_count_neurons_groups_and_folds_the_rest_into_other(tiny_atlas):
    out = tools.count_neurons(tiny_atlas, "cell_type", top=1)
    assert out["total"] == 63
    assert out["counts"] == {"PEN_a": 60}
    assert out["other"] == 3


def test_count_neurons_bad_field_lists_allowed(tiny_atlas):
    with pytest.raises(ValueError, match="top_nt"):
        tools.count_neurons(tiny_atlas, "color")


def test_describe_neuron_is_a_plain_dict(tiny_atlas):
    assert tools.describe_neuron(tiny_atlas, 100)["cell_type"] == "EPG"


def test_listing_a_selection_never_returns_more_than_50(tiny_atlas, store):
    sid = tools.search_neurons(tiny_atlas, store)["selection_id"]
    out = tools.list_selection(tiny_atlas, store, sid, limit=1000)
    assert len(out["neurons"]) == 50


def test_listing_pages_through_the_ticket(tiny_atlas, store):
    sid = tools.search_neurons(tiny_atlas, store)["selection_id"]
    out = tools.list_selection(tiny_atlas, store, sid, offset=60, limit=20)
    assert out["total"] == 63 and out["offset"] == 60 and len(out["neurons"]) == 3


def test_tool_outputs_are_plain_json(tiny_atlas, store):
    s = tools.search_neurons(tiny_atlas, store, nt="acetylcholine")
    json.dumps(s)
    json.dumps(tools.count_neurons(tiny_atlas, "side"))
    json.dumps(tools.describe_neuron(tiny_atlas, 100))
    json.dumps(tools.list_selection(tiny_atlas, store, s["selection_id"]))
