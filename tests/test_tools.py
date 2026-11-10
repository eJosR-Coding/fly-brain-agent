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


# --- hardening (H1) ---------------------------------------------------------------

def test_empty_string_filters_mean_no_filter_and_are_not_echoed(tiny_atlas, store):
    out = tools.search_neurons(tiny_atlas, store, cell_type="", nt="  ", side=" left ")
    assert out["filters"] == {"side": "left"}
    assert out["total"] == 62


def test_search_ticket_remembers_its_filters(tiny_atlas, store):
    out = tools.search_neurons(tiny_atlas, store, nt="glutamate", side="right")
    assert store.get(out["selection_id"]).filters == {"nt": "glutamate", "side": "right"}


def test_overlong_filter_is_rejected_not_echoed(tiny_atlas, store):
    with pytest.raises(ValueError, match="cell_type is too long"):
        tools.search_neurons(tiny_atlas, store, cell_type="x" * 5000)


def test_non_string_filter_is_a_type_error(tiny_atlas, store):
    with pytest.raises(TypeError, match="nt must be a string"):
        tools.search_neurons(tiny_atlas, store, nt=3)


def test_no_match_hint_lists_valid_values_of_the_bad_closed_field(tiny_atlas, store):
    out = tools.search_neurons(tiny_atlas, store, nt="dopamine")
    assert "acetylcholine" in out["hint"] and "glutamate" in out["hint"]
    assert "nt" in out["hint"]
    out = tools.search_neurons(tiny_atlas, store, side="middle")
    assert "left, right" in out["hint"]
    out = tools.search_neurons(tiny_atlas, store, super_class="nope")
    assert "central, optic, sensory" in out["hint"]


def test_no_match_hint_for_cell_type_stays_generic(tiny_atlas, store):
    out = tools.search_neurons(tiny_atlas, store, cell_type="zzz")
    assert "prefix" in out["hint"]


def test_valid_values_but_empty_combination_gets_generic_hint(tiny_atlas, store):
    out = tools.search_neurons(tiny_atlas, store, nt="glutamate", side="left")
    assert out["total"] == 0 and out["hint"]


def test_count_neurons_top_is_capped_with_a_note(tiny_atlas):
    out = tools.count_neurons(tiny_atlas, "cell_type", top=10_000)
    assert out["note"] == "top capped to 100"


def test_count_neurons_negative_top_is_clamped_with_a_note(tiny_atlas):
    out = tools.count_neurons(tiny_atlas, "cell_type", top=-3)
    assert out["counts"] == {} and out["note"] == "top -3 clamped to 0"


def test_count_neurons_in_range_top_has_no_note(tiny_atlas):
    assert "note" not in tools.count_neurons(tiny_atlas, "cell_type", top=2)


def test_count_neurons_filters_are_cleaned(tiny_atlas):
    out = tools.count_neurons(tiny_atlas, "side", nt=" acetylcholine ", cell_type="")
    assert out["total"] == 61


def test_count_neurons_non_str_group_by_is_clear(tiny_atlas):
    with pytest.raises(ValueError, match="unknown field"):
        tools.count_neurons(tiny_atlas, ["side"])


def test_list_selection_caps_limit_with_a_note(tiny_atlas, store):
    sid = tools.search_neurons(tiny_atlas, store)["selection_id"]
    out = tools.list_selection(tiny_atlas, store, sid, limit=1000)
    assert out["note"] == "limit capped to 50"


def test_list_selection_clamps_negative_offset_and_limit(tiny_atlas, store):
    sid = tools.search_neurons(tiny_atlas, store)["selection_id"]
    out = tools.list_selection(tiny_atlas, store, sid, offset=-5, limit=-1)
    assert out["offset"] == 0 and out["neurons"] == []
    assert out["note"] == "offset -5 clamped to 0; limit -1 clamped to 0"


def test_list_selection_flags_offset_past_the_end(tiny_atlas, store):
    sid = tools.search_neurons(tiny_atlas, store)["selection_id"]
    out = tools.list_selection(tiny_atlas, store, sid, offset=900)
    assert out["neurons"] == []
    assert out["note"] == "offset 900 is past the end (total 63)"


def test_list_selection_in_range_has_no_note(tiny_atlas, store):
    sid = tools.search_neurons(tiny_atlas, store)["selection_id"]
    assert "note" not in tools.list_selection(tiny_atlas, store, sid, offset=0, limit=5)


@pytest.mark.parametrize("bad", ["5", 2.0, True, None, float("nan"), [1]])
def test_paging_args_must_be_real_ints(tiny_atlas, store, bad):
    sid = tools.search_neurons(tiny_atlas, store)["selection_id"]
    with pytest.raises(TypeError, match="offset must be an int"):
        tools.list_selection(tiny_atlas, store, sid, offset=bad)
    with pytest.raises(TypeError, match="limit must be an int"):
        tools.list_selection(tiny_atlas, store, sid, limit=bad)
    with pytest.raises(TypeError, match="top must be an int"):
        tools.count_neurons(tiny_atlas, "side", top=bad)


def test_selection_id_must_be_a_string(tiny_atlas, store):
    with pytest.raises(TypeError, match="selection_id must be a string"):
        tools.list_selection(tiny_atlas, store, 5)


def test_describe_neuron_unknown_id_is_readable(tiny_atlas):
    with pytest.raises(KeyError) as exc:
        tools.describe_neuron(tiny_atlas, 9999)
    assert str(exc.value) == (
        "no neuron with root_id 9999; get ids from search_neurons or list_selection"
    )


def test_describe_neuron_rejects_float_ids(tiny_atlas):
    with pytest.raises(TypeError, match="root_id must be an int"):
        tools.describe_neuron(tiny_atlas, 100.0)
