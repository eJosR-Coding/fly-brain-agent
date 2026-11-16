"""Schema and provenance checks over every packaged preset."""

from pathlib import Path

import pytest

from flybrain import tools
from test_stimulate import store  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]
ATLAS_TSV = ROOT / "data/raw/flywire_neuron_annotations.tsv"
ROLES = {"sugar": "stimulus", "bitter": "stimulus", "water": "stimulus", "jo_ce": "stimulus",
         "mn9": "readout", "abn1": "readout", "dnp01": "readout"}
VALIDATED = {"sugar", "bitter", "water", "mn9", "jo_ce", "abn1"}
# sugar still lists its one v630 id that is gone in v783 (named in its `source`)
ABSENT_IN_V783 = {"sugar": 1}  # kept from the original preset (20 of 21 exist); the others list only present ids


def test_the_expected_presets_are_packaged():
    assert set(tools.load_presets()) == set(ROLES)


@pytest.mark.parametrize("name", sorted(ROLES))
def test_every_preset_has_a_complete_schema(name):
    p = tools.load_presets()[name]
    assert p["root_ids"] and all(isinstance(i, int) and i > 0 for i in p["root_ids"])
    assert len(set(p["root_ids"])) == len(p["root_ids"])
    assert p["description"].strip() and p["source"].strip()
    assert p["role"] == ROLES[name]
    assert isinstance(p["meaning"], str) and p["meaning"].strip()
    assert p["validation"] == ("validated" if name in VALIDATED else "exploratory")
    if "readout" in p:
        assert p["readout"] in ROLES and ROLES[p["readout"]] == "readout"


def test_dnp01_is_exploratory_and_says_why():
    p = tools.load_presets()["dnp01"]
    assert "looming selectivity was not shown" in p["source"]
    assert p["meaning"] == "escape take-off (giant fiber)"


def test_preset_selection_returns_role_meaning_validation(tiny_atlas, store):
    presets = {"orn": {"description": "d", "root_ids": [102], "source": "s",
                       "role": "stimulus", "meaning": "smell", "validation": "validated"}}
    out = tools.preset_selection(tiny_atlas, store, "orn", presets=presets)
    assert (out["role"], out["meaning"], out["validation"]) == ("stimulus", "smell", "validated")


def test_preset_selection_omits_fields_the_preset_does_not_declare(tiny_atlas, store):
    presets = {"orn": {"description": "d", "root_ids": [102], "source": "s"}}
    out = tools.preset_selection(tiny_atlas, store, "orn", presets=presets)
    assert not {"role", "meaning", "validation"} & set(out)


@pytest.mark.skipif(not ATLAS_TSV.exists(), reason="data/raw atlas not downloaded")
def test_every_preset_id_exists_in_the_real_atlas():
    import csv

    with ATLAS_TSV.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [(int(r["root_id"]), r["cell_type"]) for r in reader]
    ids = {i for i, _ in rows}
    for name, p in tools.load_presets().items():
        present = sum(1 for i in p["root_ids"] if i in ids)
        assert present == len(p["root_ids"]) - ABSENT_IN_V783.get(name, 0), name
    dnp01 = {i for i, ct in rows if ct == "DNp01"}
    assert len(dnp01) == 2 and set(tools.load_presets()["dnp01"]["root_ids"]) == dnp01
