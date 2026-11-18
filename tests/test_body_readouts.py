"""B3: per-side descending-neuron traces computed from a recording's frames (tiny hand-built fixture)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from flybrain import body_readouts as br

# atlas rows: 0 DNg100 L, 1 DNg100 R, 2 DNa02 L, 3 DNa02 R, 4 DNa01 L, 5 DNa01 R, 6 DNp01 L, 7 DNp01 R, 8 other
ATLAS = SimpleNamespace(neurons=pd.DataFrame({
    "cell_type": ["DNg100", "DNg100", "DNa02", "DNa02", "DNa01", "DNa01", "DNp01", "DNp01", "ORN"],
    "side": ["left", "right", "left", "right", "left", "right", "left", "right", "left"],
}))


def record(rows, frames, bin_ms=100.0, trials=2):
    return SimpleNamespace(rows=np.asarray(rows), frames=np.asarray(frames, dtype=np.uint8), bin_ms=bin_ms,
                           trials=trials)


def by_name(out):
    return {r["name"]: r for r in out}


def test_groups_use_exact_cell_type_and_side():
    g = br.dn_group_rows(ATLAS)
    assert g["dng100"].tolist() == [0, 1] and g["dna02_left"].tolist() == [2] and g["dna02_right"].tolist() == [3]
    assert g["dna01_left"].tolist() == [4] and g["dna01_right"].tolist() == [5] and g["dnp01"].tolist() == [6, 7]


def test_rates_are_spikes_per_neuron_trial_and_second():
    # DNg100: both neurons, 4 spikes in the first bin: 4 / (2 neurons * 2 trials * 0.1 s) = 10 Hz
    out = by_name(br.dn_readouts(record([0, 1, 2], [[2, 0], [2, 0], [0, 6]]), ATLAS))
    assert out["dng100"]["hz"] == [10.0, 0.0]
    assert out["dna02_left"]["hz"] == [0.0, 30.0]  # 6 / (1 * 2 * 0.1)
    assert out["dna02_right"]["hz"] == [0.0, 0.0] and out["dna02_right"]["neurons"] == 1
    assert out["dng100"]["name"] == "dng100" and out["dng100"]["neurons"] == 2


def test_silent_groups_are_zero_traces_and_missing_cell_types_are_skipped():
    out = br.dn_readouts(record([8], [[5, 5]]), ATLAS)  # only a non-DN neuron spiked
    assert {r["name"] for r in out} == set(br.DN_GROUPS) and all(not any(r["hz"]) for r in out)
    no_dn = SimpleNamespace(neurons=pd.DataFrame({"cell_type": ["ORN"], "side": ["left"]}))
    assert br.dn_readouts(record([0], [[1]]), no_dn) == []


def test_the_traces_feed_the_drive_with_the_right_turn_sign():
    from flybrain import body_drive as bd

    out = br.dn_readouts(record([0, 1, 2], [[20] * 10] * 3, bin_ms=50.0, trials=1), ATLAS)
    d = bd.drive_from_readouts(out, bin_ms=50.0, smooth_tau_ms=0.0)
    assert (d.forward > 0).all() and (d.turn > 0).all()  # DNa02 LEFT firing, right silent: turn left
    assert d.drive[0, 0] < d.drive[0, 1]


def test_empty_recording_gives_no_bins():
    out = br.dn_readouts(record(np.zeros(0, int), np.zeros((0, 4))), ATLAS)
    assert all(r["hz"] == [0.0] * 4 for r in out)


def test_body_from_record_drives_the_runner_with_the_per_side_drive(monkeypatch):
    from flybrain import body_runner, body_service

    seen = {}

    def fake(drive, bin_ms, duration_ms, seed=0, escape=None):
        seen.update(drive=drive, bin_ms=bin_ms, duration_ms=duration_ms, seed=seed)
        return {"format": "fly-walk/1"}

    monkeypatch.setattr(body_runner, "run_body", fake)
    rec = SimpleNamespace(rows=np.array([0, 1, 2]), frames=np.full((3, 10), 20, np.uint8), bin_ms=50.0, trials=1,
                          duration_ms=500.0)
    out = body_service.body_from_record(rec, ATLAS, duration_ms=300, seed=3)
    assert seen["duration_ms"] == 300 and seen["seed"] == 3 and seen["bin_ms"] == 50.0
    assert (seen["drive"].turn > 0).all() and out["drive"]["turn"][-1] > 0 and out["drive"]["duration_ms"] == 300
    names = {r["name"] for r in out["drive"]["readouts"]}
    assert {"dng100", "dna02_left", "dna02_right"} <= names


def test_effective_duration_rejects_non_positive_and_caps():
    from flybrain import body_service

    rec = SimpleNamespace(duration_ms=5000.0)
    assert body_service.effective_duration_ms(rec, None) == body_service.MAX_BODY_MS
    assert body_service.effective_duration_ms(rec, 800) == 800
    for bad in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            body_service.effective_duration_ms(rec, bad)
