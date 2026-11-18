"""looming_escape(input="motion") over a tiny T4/T5 eye + network."""

import json

import numpy as np
import pytest

from flybrain import eye, looming
from flybrain.atlas import load_atlas
from flybrain.sim_service import SimulationService
from flybrain.simulation import Simulation

HEADER = "root_id\tpos_x\tpos_y\tpos_z\tsuper_class\tcell_class\tcell_type\ttop_nt\tside\tflow"
TYPES = [f"T{g}{k}" for g in (4, 5) for k in "abcd"]
GRID = [(h, v) for h in range(-3, 4) for v in range(-3, 4) if (h, v) != (0, 0)]  # 48 cells per side
N_MOTION = len(GRID)
LPLC2, LC4, DNP01, DNP04 = [96, 97, 98], [99, 100, 101], [102], [103]


@pytest.fixture
def atlas(tmp_path):
    lines = [HEADER]
    for side, x0 in (("right", 700000), ("left", 300000)):
        for i, (h, v) in enumerate(GRID):
            lines.append(f"{len(lines)}\t{x0}\t{-v * 1000}\t{h * 1000}\toptic\t\t{TYPES[(i * 3 + i // 8) % 8]}\tacetylcholine\t{side}\tintrinsic")
    for t, ids in (("LPLC2", LPLC2), ("LC4", LC4), ("DNp01", DNP01), ("DNp04", DNP04)):
        for k, _ in enumerate(ids):
            lines.append(f"{len(lines)}\t700000\t{k * 1000}\t{k * 2000}\toptic\t\t{t}\tacetylcholine\tright\tafferent")
    p = tmp_path / "a.tsv"
    p.write_text("\n".join(lines) + "\n")
    return load_atlas(p)


@pytest.fixture
def sims(atlas):
    edges = [(r, d, 1000) for r in range(0, 2 * N_MOTION) for d in LPLC2 + LC4]
    edges += [(d, t, 1000) for d in LPLC2 + LC4 for t in DNP01 + DNP04]
    pre, post, w = (np.array(c) for c in zip(*edges))
    return SimulationService.from_simulation(Simulation.from_arrays(pre, post, w, n_neurons=len(atlas.neurons)))


@pytest.fixture
def calls(sims, monkeypatch):
    seen = []
    real = sims.run_schedule

    def spy(schedule, duration_ms=1000.0, trials=1, seed=0, silence_rows=None, bin_ms=None):
        seen.append((schedule, None if silence_rows is None else sorted(int(r) for r in silence_rows)))
        return real(schedule, duration_ms, trials, seed=seed, silence_rows=silence_rows, bin_ms=bin_ms)

    monkeypatch.setattr(sims, "run_schedule", spy)
    return seen


def run(atlas, sims, **kw):
    opts = {"input": "motion", "rate_hz": 300, "duration_ms": 300, "trials": 3, "steps": 6, **kw}
    return looming.looming_escape(atlas, sims, **opts)


def test_motion_is_a_valid_input_and_unknown_ones_are_still_refused(atlas, sims):
    assert run(atlas, sims)["input"] == "motion"
    with pytest.raises(ValueError, match="input"):
        run(atlas, sims, input="retina")


def test_all_four_conditions_get_exactly_equal_total_drive(atlas, sims, calls):
    run(atlas, sims)
    drives = [eye.schedule_drive(s) for s, _ in calls]
    assert len(drives) == 4 and drives[0] > 0
    assert drives == pytest.approx([drives[0]] * 4, rel=1e-6)


def test_only_the_chosen_sides_t4_t5_cells_are_driven(atlas, sims, calls):
    run(atlas, sims)
    right = {i for i in range(N_MOTION)}
    for s, _ in calls:
        assert {int(r) for _, _, rows, _ in s.segments for r in rows} <= right
    calls.clear()
    run(atlas, sims, side="left")
    assert {int(r) for _, _, rows, _ in calls[0][0].segments for r in rows} <= set(range(N_MOTION, 2 * N_MOTION))


def test_dimming_is_motion_free_uniform_drive_and_control_is_shuffled_looming(atlas, sims, calls):
    run(atlas, sims)
    (start, end, rows, rate), = calls[2][0].segments
    assert (start, end) == (0.0, 300.0) and len(rows) == N_MOTION and rate >= 1.0  # every cell alike: no direction
    loom, ctrl = calls[0][0].segments, calls[3][0].segments
    assert [(a, b, len(r), hz) for a, b, r, hz in loom] == [(a, b, len(r), hz) for a, b, r, hz in ctrl]
    assert any(list(a[2]) != list(b[2]) for a, b in zip(loom, ctrl))
    assert all(sil is None for _, sil in calls)


def test_looming_and_receding_use_different_cells_by_the_dark_disc_polarity(atlas, sims, calls):
    run(atlas, sims)
    names = atlas.neurons["cell_type"].to_numpy()
    loom_types = {names[int(r)][:2] for _, _, rows, _ in calls[0][0].segments for r in rows}
    rec_types = {names[int(r)][:2] for _, _, rows, _ in calls[1][0].segments for r in rows}
    assert loom_types == {"T5"} and rec_types == {"T4"}


def test_polarity_both_drives_t4_and_t5_in_every_condition(atlas, sims, calls):
    looming.looming_escape(atlas, sims, input="motion", polarity="both", rate_hz=300, duration_ms=300, trials=2, steps=6)
    names = atlas.neurons["cell_type"].to_numpy()
    assert {names[int(r)][:2] for _, _, rows, _ in calls[0][0].segments for r in rows} == {"T4", "T5"}


def test_report_has_detector_and_dnp04_means_and_the_front_end_is_labelled_our_choice(atlas, sims):
    out = run(atlas, sims)
    loom = out["conditions"]["looming"]
    assert {"dnp01_hz", "lplc2_hz", "lc4_hz", "dnp04_hz"} <= set(loom)
    assert out["stimulus"]["neurons"] == N_MOTION and "photoreceptors" not in out
    assert "our model choice" in out["input_note"] and "Maisak" in out["note"]
    assert "model prediction" in out["verdict"]
    assert len(json.dumps(run(atlas, sims, trials=5, steps=10))) < 2800


def test_detector_input_output_is_unchanged_by_the_new_mode(atlas, sims):
    out = looming.looming_escape(atlas, sims, input="detectors", rate_hz=300, duration_ms=300, trials=2, steps=6)
    assert "dnp04_hz" not in out["conditions"]["looming"]
