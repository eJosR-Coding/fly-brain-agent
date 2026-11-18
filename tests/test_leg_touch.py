import numpy as np

from flybrain import live


def test_packaged_mapping_has_every_leg_side_and_provenance():
    import json
    d = json.loads(live.LEG_TOUCH_JSON.read_text())
    assert d["provenance"]["license"] == "CC BY 4.0" and "not validated" in d["provenance"]["method"]
    assert set(d["legs"]) == {f"{l}_{s}" for l in ("front", "middle", "hind") for s in ("left", "right")}
    assert all(v["ans"] for v in d["legs"].values())


def test_load_and_rows_head_on_front_only_else_all_legs_of_the_side():
    legs = live.load_leg_touch()
    assert set(legs["left"]) == {"front", "all"} and set(legs["left"]["front"]) <= set(legs["left"]["all"])
    rows = {"left": {"front": np.array([1, 2]), "all": np.array([1, 2, 3])}, "right": {"front": np.array([2, 9]), "all": np.array([2, 9, 8])}}
    assert live.leg_touch_rows(rows, ["left", "right"]).tolist() == [1, 2, 9]
    assert live.leg_touch_rows(rows, ["right"]).tolist() == [2, 8, 9]
    assert live.leg_touch_rows(rows, []).size == 0
