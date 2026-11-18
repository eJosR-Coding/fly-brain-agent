"""B1: the pure DN readout -> flygym descending-drive mapping (our choice, not a published one)."""

import numpy as np
import pytest

from flybrain import body_drive as bd


def ro(name, hz):
    return {"name": name, "label": name, "neurons": 2, "rows": [], "hz": list(hz)}


def run(readouts, n=10, bin_ms=50.0, **kw):
    return bd.drive_from_readouts(readouts, bin_ms=bin_ms, smooth_tau_ms=0.0, **kw)


def flat(v, n=10):
    return [v] * n


def test_forward_is_zero_below_threshold_and_saturates_above():
    low = run([ro("DNg100", flat(bd.FORWARD_THRESHOLD_HZ * 0.9))])
    assert (low.forward == 0).all() and (low.drive == 0).all()
    mid = run([ro("DNg100", flat((bd.FORWARD_THRESHOLD_HZ + bd.FORWARD_SATURATION_HZ) / 2))])
    assert 0.4 < mid.forward[0] < 0.6
    hi = run([ro("DNg100", flat(bd.FORWARD_SATURATION_HZ * 5))])
    assert (hi.forward == 1.0).all()


def test_forward_walks_straight_with_equal_sides_in_the_valid_range():
    d = run([ro("DNg100", flat(bd.FORWARD_SATURATION_HZ))])
    assert np.allclose(d.drive[:, 0], d.drive[:, 1])
    assert d.drive.min() >= bd.DRIVE_MIN and d.drive.max() <= bd.DRIVE_MAX
    weak = run([ro("DNg100", flat(bd.FORWARD_THRESHOLD_HZ + 1))])
    assert weak.drive[0, 0] < d.drive[0, 0]


def test_turn_sign_stronger_left_neurons_turn_left_by_lowering_the_left_drive():
    left = run([ro("DNg100", flat(60)), ro("DNa02_left", flat(40)), ro("DNa02_right", flat(0))])
    assert (left.turn > 0).all()  # positive = turn left (counter-clockwise seen from above)
    assert (left.drive[:, 0] < left.drive[:, 1]).all()
    right = run([ro("DNg100", flat(60)), ro("DNa02_left", flat(0)), ro("DNa02_right", flat(40))])
    assert (right.turn < 0).all() and (right.drive[:, 0] > right.drive[:, 1]).all()


def test_turn_is_a_saturating_left_minus_right_difference_and_zero_when_balanced():
    bal = run([ro("DNg100", flat(60)), ro("DNa02_left", flat(30)), ro("DNa02_right", flat(30)),
               ro("DNa01_left", flat(20)), ro("DNa01_right", flat(20))])
    assert np.allclose(bal.turn, 0)
    big = run([ro("DNg100", flat(60)), ro("DNa02_left", flat(1000)), ro("DNa02_right", flat(0))])
    assert (big.turn == 1.0).all()
    # both cell types firing on ONE side only is a turn; the same rate on both sides is not
    assert (run([ro("DNa01_left", flat(30))]).turn > 0).all()
    assert np.allclose(run([ro("DNa01_left", flat(30)), ro("DNa01_right", flat(30))]).turn, 0)


def test_dna02_has_more_steering_gain_than_dna01_and_both_add_up():
    a02 = run([ro("DNa02_left", flat(10))]).turn[0]
    a01 = run([ro("DNa01_left", flat(10))]).turn[0]
    assert a02 == pytest.approx(10 * bd.DNA02_GAIN / bd.TURN_SATURATION_HZ)
    assert 0 < a01 < a02 and bd.DNA02_GAIN > bd.DNA01_GAIN
    both = run([ro("DNa02_left", flat(10)), ro("DNa01_left", flat(10))]).turn[0]
    assert both == pytest.approx(a02 + a01)


def test_the_unsided_cell_type_names_are_not_a_turn_signal():
    d = run([ro("DNa02", flat(40)), ro("DNa01", flat(0))])
    assert (d.turn == 0).all()


def test_turn_alone_still_walks_at_a_modest_pace():
    d = run([ro("DNa02_left", flat(40))])
    assert (d.drive > 0).all() and d.drive.max() <= bd.DRIVE_MAX
    assert (d.forward == 0).all()


def test_escape_takes_precedence_stops_legs_and_flags_wings():
    d = run([ro("DNg100", flat(60)), ro("DNa02_left", flat(40)), ro("DNp01", [0, 0, 80, 80, 0, 0, 0, 0, 0, 0])])
    assert d.escape.tolist() == [False, False, True, True] + [False] * 6
    assert (d.drive[d.escape] == 0).all() and (d.drive[~d.escape] > 0).all()
    assert d.escape_strength[2] == 1.0 and d.escape_strength[0] == 0.0


def test_missing_readouts_give_zero_components():
    d = run([])
    assert d.n_bins == 0
    d = run([ro("DNa02_left", flat(40))])
    assert (d.forward == 0).all() and (d.escape == 0).all()
    d = run([ro("DNg100", flat(60))])
    assert (d.turn == 0).all()
    d = run([ro("Other", flat(60))], n=10)
    assert (d.drive == 0).all() and d.n_bins == 0


def test_names_match_case_insensitively_and_none_names_are_ignored():
    d = run([{"name": None, "label": "x", "hz": flat(99)}, ro("dng100", flat(60))])
    assert (d.forward == 1).all()


def test_unequal_lengths_use_the_longest_and_pad_with_zero():
    d = run([ro("DNg100", flat(60, 10)), ro("DNa02_left", flat(40, 4))])
    assert d.n_bins == 10 and (d.turn[:4] > 0).all() and (d.turn[4:] == 0).all()


def test_drive_stays_in_range_for_extreme_inputs():
    d = run([ro("DNg100", flat(1e9)), ro("DNa02_left", flat(1e9)), ro("DNa02_right", flat(0))])
    assert d.drive.min() >= bd.DRIVE_MIN and d.drive.max() <= bd.DRIVE_MAX


def test_smoothing_is_a_causal_low_pass_over_bins():
    spike = [0, 0, 100, 0, 0, 0, 0, 0, 0, 0]
    raw = bd.drive_from_readouts([ro("DNg100", spike)], bin_ms=50.0, smooth_tau_ms=0.0)
    sm = bd.drive_from_readouts([ro("DNg100", spike)], bin_ms=50.0, smooth_tau_ms=100.0)
    assert raw.forward[1] == 0 and sm.forward[1] == 0  # causal: nothing before the spike
    assert sm.forward[2] < raw.forward[2] and sm.forward[3] > raw.forward[3]


def test_deterministic_and_does_not_mutate_inputs():
    r = [ro("DNg100", flat(30)), ro("DNa02_left", flat(20)), ro("DNp01", flat(0))]
    import copy
    before = copy.deepcopy(r)
    a, b = bd.drive_from_readouts(r, bin_ms=50.0), bd.drive_from_readouts(r, bin_ms=50.0)
    assert np.array_equal(a.drive, b.drive) and r == before


def test_rejects_bad_bin_ms_and_has_an_honest_caption():
    with pytest.raises(ValueError):
        bd.drive_from_readouts([], bin_ms=0)
    assert "our" in bd.CAPTION.lower() and "not published" in bd.CAPTION.lower()
    assert run([ro("DNg100", flat(60))]).caption == bd.CAPTION


def test_as_dict_is_json_serialisable():
    import json
    d = run([ro("DNg100", flat(60))])
    json.dumps(d.as_dict())
