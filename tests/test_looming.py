"""looming_escape over a tiny hand-built eye + network (no real weights, no LLM)."""

import json

import numpy as np
import pytest

from flybrain import eye, looming
from flybrain.atlas import load_atlas
from flybrain.sim_service import SimulationService
from flybrain.simulation import Simulation

HEADER = "root_id\tpos_x\tpos_y\tpos_z\tsuper_class\tcell_class\tcell_type\ttop_nt\tside\tflow"
# rows: 0-24 left eye, 25-49 right eye (R1-6 on a 5x5 grid), 50-52 LPLC2, 53-55 LC4 (right, on a line),
# 56 DNp01, 57-58 LPLC2 and 59-60 LC4 on the left side
LPLC2, LC4, DNP01 = [50, 51, 52], [53, 54, 55], [56]
LEFT_DET = [57, 58, 59, 60]
RIGHT = list(range(25, 50))
INHIBITORY = RIGHT[:5]  # 5 of the 25 right-eye cells have negative outgoing weights


@pytest.fixture
def atlas(tmp_path):
    lines = [HEADER]
    for i in range(50):
        side = "left" if i < 25 else "right"
        x, y = (i % 25) % 5, (i % 25) // 5
        lines.append(f"{i + 1}\t{x * 1000 + (0 if side == 'left' else 100000)}\t{y * 1000}\t0\tsensory\t\tR1-6\thistamine\t{side}\tafferent")
    for i, t in zip(range(50, 61), ["LPLC2"] * 3 + ["LC4"] * 3 + ["DNp01"] + ["LPLC2"] * 2 + ["LC4"] * 2):
        side = "left" if i >= 57 else "right"
        lines.append(f"{i + 1}\t{(i - 50) * 1000}\t500\t0\toptic\t\t{t}\tacetylcholine\t{side}\tafferent")
    p = tmp_path / "a.tsv"
    p.write_text("\n".join(lines) + "\n")
    return load_atlas(p)


@pytest.fixture
def sims(atlas):
    edges = []
    for r in range(25, 50):  # right eye drives both detectors
        w = -1000 if r in INHIBITORY else 1000
        edges += [(r, d, w) for d in LPLC2 + LC4]
    edges += [(d, 56, 1000) for d in LPLC2 + LC4]
    pre, post, w = (np.array(c) for c in zip(*edges))
    return SimulationService.from_simulation(Simulation.from_arrays(pre, post, w, n_neurons=len(atlas.neurons)))


@pytest.fixture
def calls(sims, monkeypatch):
    """Every run_schedule call, in order: (schedule, silence_rows)."""
    seen = []
    real = sims.run_schedule

    def spy(schedule, duration_ms=1000.0, trials=1, seed=0, silence_rows=None, bin_ms=None):
        seen.append((schedule, None if silence_rows is None else sorted(int(r) for r in silence_rows)))
        return real(schedule, duration_ms, trials, seed=seed, silence_rows=silence_rows, bin_ms=bin_ms)

    monkeypatch.setattr(sims, "run_schedule", spy)
    return seen


def run(atlas, sims, **kw):
    opts = {"input": "photoreceptors", "rate_hz": 300, "duration_ms": 300, "trials": 4, "steps": 6, **kw}
    return looming.looming_escape(atlas, sims, **opts)


def test_all_conditions_get_exactly_equal_input_drive(atlas, sims, calls):
    run(atlas, sims)
    drives = [eye.schedule_drive(s) for s, _ in calls]
    assert len(drives) == 4 and drives[0] > 0
    assert drives == pytest.approx([drives[0]] * 4, rel=1e-12)


def test_receding_is_the_time_reverse_of_looming(atlas, sims, calls):
    run(atlas, sims, duration_ms=300)
    loom, rec = calls[0][0], calls[1][0]
    mirrored = sorted((300 - end, 300 - start, rows, rate) for start, end, rows, rate in loom.segments)
    assert mirrored == sorted(rec.segments) and loom.segments != rec.segments


def test_dimming_drives_the_whole_eye_for_the_whole_run_at_one_rate(atlas, sims, calls):
    out = run(atlas, sims)
    (start, end, rows, rate), = calls[2][0].segments
    assert (start, end) == (0.0, 300.0) and list(rows) == RIGHT and 1.0 <= rate < 300
    assert out["drive"]["dimming_rate_hz"] == pytest.approx(rate, abs=0.05)


def test_dimming_below_the_rate_floor_is_shortened_instead(atlas, sims, calls):
    out = run(atlas, sims, rate_hz=2)
    (start, end, rows, rate), = calls[2][0].segments
    assert rate == 1.0 and start == 0.0 and end < 300.0
    drives = [eye.schedule_drive(s) for s, _ in calls]
    assert drives[2] == pytest.approx(drives[0], rel=0.01)  # on the 0.1 ms grid, not bit-exact
    assert "shorter" in out["note"]


def test_control_silences_lplc2_and_lc4_and_nothing_else(atlas, sims, calls):
    run(atlas, sims)
    assert [s for _, s in calls[:3]] == [None, None, None]
    assert calls[3][1] == sorted(LPLC2 + LC4 + LEFT_DET)
    assert calls[3][0].segments == calls[0][0].segments


def test_results_have_per_condition_readouts_and_a_looming_trace(atlas, sims):
    out = run(atlas, sims)
    assert out["experiment"] == "looming_escape"
    assert set(out["conditions"]) == {"looming", "receding", "dimming", "control"}
    loom = out["conditions"]["looming"]
    assert loom["dnp01_hz"] > 0 and loom["lplc2_hz"] > 0 and loom["lc4_hz"] > 0
    assert loom["ci_low"] <= loom["dnp01_hz"] <= loom["ci_high"]
    ctrl = out["conditions"]["control"]
    assert ctrl["dnp01_hz"] == 0 and ctrl["lplc2_hz"] > 0  # detectors still fire, their output is cut
    assert len(out["looming_dnp01_trace_hz"]) == 6 and max(out["looming_dnp01_trace_hz"]) > 0


def test_stats_compare_looming_with_each_other_condition_and_holm_correct(atlas, sims):
    out = run(atlas, sims)
    comps = {c["vs"]: c for c in out["comparisons"]}
    assert list(comps) == ["receding", "dimming", "control"]
    ctrl = comps["control"]
    assert ctrl["delta_hz"] > 0 and ctrl["significant"] is True and ctrl["p_adj"] >= ctrl["p"]
    for c in comps.values():
        assert c["p_adj"] is None or c["p_adj"] >= c["p"]
        assert {"delta_hz", "ci_low", "ci_high", "p", "p_adj", "cohens_d", "significant"} <= set(c)
    assert "model prediction" in out["verdict"]


def test_no_significant_difference_is_said_plainly(atlas, sims):
    out = run(atlas, sims, trials=1)
    assert all(c["p"] is None and c["significant"] is False for c in out["comparisons"])
    assert "at least 2 trials" in out["verdict"]


def test_photoreceptor_sign_share_is_reported(atlas, sims):
    out = run(atlas, sims)
    assert out["photoreceptors"] == {"eye": "right", "n": 25, "excitatory_pct": 80.0, "inhibitory_pct": 20.0}
    assert "histamine" in out["note"] and "T4" in out["note"] and "model prediction" in out["note"]


def test_recordings_are_stored_for_the_viewer(atlas, sims):
    out = run(atlas, sims)
    ids = {k: v["simulation_id"] for k, v in out["conditions"].items()}
    assert len(set(ids.values())) == 4
    rec = sims.get(ids["looming"])
    assert rec.readout["label"] == "DNp01" and rec.trials == 4 and rec.active_count > 25
    assert max(rec.readout["hz"]) > 0 and len(rec.readout["hz"]) == rec.frames.shape[1]
    assert list(sims.get(ids["control"]).silenced_rows) == sorted(LPLC2 + LC4 + LEFT_DET)
    assert out["simulation_id"] == ids["looming"]


def test_output_stays_small(atlas, sims):
    assert len(json.dumps(run(atlas, sims, trials=5, steps=10))) < 2500


def test_parameters_are_clamped_and_validated(atlas, sims):
    out = run(atlas, sims, rate_hz=9999, trials=99, steps=500, duration_ms=5)
    s = out["stimulus"]
    assert s["rate_hz"] == 300 and s["trials"] == 10 and s["steps"] == 20 and s["duration_ms"] == 100
    assert "capped" in out["note"] and "clamped" in out["note"]
    with pytest.raises(ValueError, match="side"):
        run(atlas, sims, side="up")
    with pytest.raises(TypeError, match="rate_hz"):
        run(atlas, sims, rate_hz="fast")


# ---- input="detectors": the disc is injected at LPLC2 + LC4 instead of the photoreceptors ----

def test_detector_input_drives_only_the_chosen_sides_lplc2_and_lc4(atlas, sims, calls):
    run(atlas, sims, input="detectors")
    loom = calls[0][0]
    assert {int(r) for _, _, rows, _ in loom.segments for r in rows} <= set(LPLC2 + LC4)
    assert list(calls[2][0].segments[0][2]) == sorted(LPLC2 + LC4)  # dimming: all detectors
    calls.clear()
    run(atlas, sims, input="detectors", side="left")
    assert {int(r) for _, _, rows, _ in calls[0][0].segments for r in rows} <= set(LEFT_DET)


def test_detector_input_keeps_equal_drive_and_does_not_silence_anything(atlas, sims, calls):
    run(atlas, sims, input="detectors")
    drives = [eye.schedule_drive(s) for s, _ in calls]
    assert len(drives) == 4 and drives[0] > 0 and drives == pytest.approx([drives[0]] * 4, rel=0.01)
    assert [s for _, s in calls] == [None] * 4


def test_shuffled_control_has_the_same_counts_per_window_but_other_rows(atlas, sims, calls):
    run(atlas, sims, input="detectors")
    loom, ctrl = calls[0][0].segments, calls[3][0].segments
    assert [(a, b, len(r), hz) for a, b, r, hz in loom] == [(a, b, len(r), hz) for a, b, r, hz in ctrl]
    assert any(list(a[2]) != list(b[2]) for a, b in zip(loom, ctrl))
    assert all(set(map(int, r)) <= set(LPLC2 + LC4) and len(set(map(int, r))) == len(r) for _, _, r, _ in ctrl)


def test_shuffled_control_is_reproducible(atlas, sims, calls):
    run(atlas, sims, input="detectors")
    first = [list(r) for _, _, r, _ in calls[3][0].segments]
    calls.clear()
    run(atlas, sims, input="detectors")
    assert [list(r) for _, _, r, _ in calls[3][0].segments] == first


def test_detector_output_reports_input_and_note(atlas, sims):
    out = run(atlas, sims, input="detectors")
    assert out["input"] == "detectors" and "halfway" in out["input_note"]
    assert out["conditions"]["looming"]["dnp01_hz"] > 0
    assert "photoreceptors" not in out and out["stimulus"]["neurons"] == 6
    assert "SHUFFLED" in out["note"]
    assert len(json.dumps(run(atlas, sims, input="detectors", trials=5, steps=10))) < 2500


def test_photoreceptor_output_cites_the_model_limit(atlas, sims):
    out = run(atlas, sims)
    assert out["input"] == "photoreceptors" and "Shiu et al. 2024" in out["input_note"]


def test_default_input_is_detectors(atlas, sims):
    out = looming.looming_escape(atlas, sims, rate_hz=300, duration_ms=300, trials=2, steps=6)
    assert out["input"] == "detectors"


def test_unknown_input_is_a_clear_error(atlas, sims):
    with pytest.raises(ValueError, match="input"):
        run(atlas, sims, input="retina")
