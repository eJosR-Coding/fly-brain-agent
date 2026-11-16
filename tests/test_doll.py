"""The doll's pure state: what is driven, silenced and watched, and how a call changes it."""

import pytest

from flybrain.doll import MAX_STIMULI, Doll, describe_change


def test_a_new_doll_is_empty():
    d = Doll()
    assert d.conditions() == {"stimuli": {}, "silenced": [], "readouts": []}
    assert d.last is None and not d.has_stimulus()


def test_add_then_replace_a_stimulus():
    d = Doll()
    d.apply(stimuli={"sugar": 100})
    d.apply(stimuli={"bitter": 50})
    assert d.conditions()["stimuli"] == {"sugar": 100.0, "bitter": 50.0}
    d.apply(stimuli={"sugar": 20})  # same name: the new rate replaces
    assert d.conditions()["stimuli"] == {"sugar": 20.0, "bitter": 50.0}


def test_remove_drops_from_stimuli_silenced_and_readouts_and_notes_unknowns():
    d = Doll()
    d.apply(stimuli={"sugar": 100, "bitter": 100}, silence=["dnp01"], readouts=["mn9"])
    notes = d.apply(remove=["bitter", "dnp01", "mn9", "ghost"])
    assert d.conditions() == {"stimuli": {"sugar": 100.0}, "silenced": [], "readouts": []}
    assert len(notes) == 1 and "ghost" in notes[0]


def test_silence_and_readouts_persist_and_do_not_duplicate():
    d = Doll()
    d.apply(silence=["a", "b"], readouts=["mn9"])
    d.apply(silence=["a"], readouts=["mn9", "abn1"])
    assert d.conditions()["silenced"] == ["a", "b"] and d.conditions()["readouts"] == ["mn9", "abn1"]


def test_reset_clears_first_then_applies_the_rest_of_the_call():
    d = Doll()
    d.apply(stimuli={"sugar": 100}, silence=["a"], readouts=["mn9"])
    d.last = {"active_neurons": 1}
    d.apply(reset=True, stimuli={"jo_ce": 80})
    assert d.conditions() == {"stimuli": {"jo_ce": 80.0}, "silenced": [], "readouts": []}
    assert d.last == {"active_neurons": 1}  # reset changes the conditions, the previous run stays the baseline


def test_order_is_reset_then_remove_then_set():
    d = Doll()
    d.apply(stimuli={"sugar": 100})
    d.apply(reset=True, remove=["sugar"], stimuli={"sugar": 30})  # removed first, then set again
    assert d.conditions()["stimuli"] == {"sugar": 30.0}
    d.apply(remove=["sugar"], stimuli={"sugar": 40})
    assert d.conditions()["stimuli"] == {"sugar": 40.0}


def test_at_most_six_stimuli_and_a_failed_call_changes_nothing():
    d = Doll()
    d.apply(stimuli={f"s{i}": 10 for i in range(MAX_STIMULI)})
    before = d.conditions()
    with pytest.raises(ValueError, match="at most 6"):
        d.apply(stimuli={"extra": 10}, silence=["x"])
    assert d.conditions() == before
    d.apply(stimuli={"s0": 99})  # replacing is not adding
    assert d.conditions()["stimuli"]["s0"] == 99.0


def test_conditions_is_a_copy():
    d = Doll()
    d.apply(stimuli={"sugar": 100})
    d.conditions()["stimuli"]["sugar"] = 1
    assert d.conditions()["stimuli"]["sugar"] == 100.0


def test_describe_change_names_what_moved():
    a = {"stimuli": {"sugar": 100.0}, "silenced": [], "readouts": []}
    b = {"stimuli": {"sugar": 100.0, "bitter": 100.0}, "silenced": [], "readouts": []}
    assert describe_change(a, b) == "added bitter 100 Hz"
    assert describe_change(b, a) == "removed bitter"
    c = {"stimuli": {"sugar": 50.0}, "silenced": ["dnp01"], "readouts": ["mn9"]}
    text = describe_change(a, c)
    assert "sugar 100 -> 50 Hz" in text and "silenced dnp01" in text and "watching mn9" in text
    assert describe_change(c, a).count("unsilenced dnp01") == 1 and "stopped watching mn9" in describe_change(c, a)
    assert describe_change(a, a) == "no change in conditions"
