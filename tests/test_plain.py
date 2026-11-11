"""Plain-language layer: a glossary plus a template, so every neuron gets a sentence a non-scientist can read."""

import json
from importlib import resources

import pytest

from flybrain.plain import glossary, plain_description

# every cell_class value in the FlyWire annotations (v783)
ALL_CELL_CLASSES = """ME>LO ME visual LA>ME ME>LOP LO>LOP ME>LO.LOP Kenyon_Cell ME>LA CX mechanosensory olfactory AN LO
LOP>ME.LO LOP ALPN LOP>LO LHLN ALLN ME>LOP.LO gustatory DAN LA LO>ME bilateral LOP>LO.ME ME.LO TuBu unknown_sensory
brain_motor_neuron MBON mAL hygrosensory ocellar pars_intercerebralis LHCENT optic_lobes ME.LO.LOP pars_lateralis
thermosensory ALIN ALON LOP>ME MBIN ME.LOP TPN LO.LOP neck_motor_neuron""".split()
ALL_SUPER = ["optic", "central", "sensory", "visual_projection", "ascending", "descending",
             "sensory_ascending", "visual_centrifugal", "motor", "endocrine"]
ALL_NT = ["acetylcholine", "glutamate", "gaba", "dopamine", "serotonin", "octopamine"]


def info(**kw):
    return {"cell_type": None, "cell_class": None, "super_class": None, "top_nt": None, **kw}


def test_glossary_is_packaged_json_with_every_section():
    text = resources.files("flybrain").joinpath("data/glossary.json").read_text(encoding="utf-8")
    assert {"super_class", "cell_class", "neuropils", "nt", "families"} <= set(json.loads(text))


@pytest.mark.parametrize("cls", ALL_CELL_CLASSES)
def test_every_real_cell_class_gets_a_one_sentence_description(cls):
    out = plain_description(info(cell_class=cls, super_class="optic"))
    first = out.split(". ")[0]
    assert len(first) > 20 and "None" not in out


def test_every_super_class_and_transmitter_is_covered():
    g = glossary()
    assert set(ALL_SUPER) <= set(g["super_class"]) and set(ALL_NT) <= set(g["nt"])


def test_optic_lobe_class_is_built_from_neuropil_names():
    assert "medulla" in plain_description(info(cell_class="ME>LO.LOP", super_class="optic"))
    out = plain_description(info(cell_class="LA>ME", super_class="optic"))
    assert "lamina" in out and "medulla" in out
    assert "lobula plate" in plain_description(info(cell_class="LOP", super_class="optic"))
    assert "lobula plate" in plain_description(info(cell_class="LO.LOP", super_class="optic"))


def test_famous_types_win_over_class_and_template():
    assert "proboscis" in plain_description(info(cell_type="CB0701", cell_class="brain_motor_neuron", super_class="motor"))
    assert "escape" in plain_description(info(cell_type="DNp01", super_class="descending"))
    assert "looming" in plain_description(info(cell_type="LPLC2", super_class="visual_projection"))
    assert "heading" in plain_description(info(cell_type="EPG", cell_class="CX", super_class="central"))
    assert "odor" in plain_description(info(cell_type="ORN_DA1", cell_class="olfactory", super_class="sensory"))
    assert "photoreceptor" in plain_description(info(cell_type="R7", cell_class="visual", super_class="sensory")).lower()


def test_unclassed_neuron_uses_the_super_class_transmitter_and_output_template():
    out = plain_description(info(cell_type="CB0328", super_class="central", top_nt="glutamate"),
                            output_super_class="descending")
    assert out.startswith("A central-brain")
    assert "glutamate" in out and "usually inhibits" in out
    assert "strongest output is a descending neuron" in out and "commands" in out


def test_transmitter_meaning_follows_the_model_rule():
    assert "inhibits" in plain_description(info(super_class="central", top_nt="gaba"))
    assert "excites" in plain_description(info(super_class="central", top_nt="acetylcholine"))
    assert "modulat" in plain_description(info(super_class="central", top_nt="dopamine"))


def test_missing_everything_still_gives_a_sentence_and_never_crashes():
    assert plain_description(info()) == "A neuron with no annotation in this atlas."
    out = plain_description(info(cell_class="never_heard_of_it", top_nt="mystery"))
    assert out and "None" not in out
