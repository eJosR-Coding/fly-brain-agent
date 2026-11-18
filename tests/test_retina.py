"""Exact retinotopy of T4/T5 from the Codex column file: lattice (p, q) -> head-frame angles, with a tiny fixture."""

import gzip

import numpy as np
import pytest

from flybrain import eye, looming, retina
from flybrain.atlas import load_atlas
from flybrain.sim_service import SimulationService
from flybrain.simulation import Simulation

HEADER = "root_id\tpos_x\tpos_y\tpos_z\tsuper_class\tcell_class\tcell_type\ttop_nt\tside\tflow"
TYPES = [f"T{g}{k}" for g in (4, 5) for k in "abcd"]
# lattice points (p, q): x = (q - p) / 2 is frontward, y = p + q is up
LATTICE = [(-4, 8), (5, -10), (0, 0), (3, 3), (-3, -3), (6, 2), (-6, -2), (2, -4), (-2, 4), (8, 0)]
LPLC2, LC4, DNP01, DNP04 = [20, 21], [22, 23], [24], [25]  # rows: 20 T4/T5 cells first


@pytest.fixture
def atlas(tmp_path):
    lines = [HEADER]
    for side in ("right", "left"):
        for i, _ in enumerate(LATTICE):
            lines.append(f"{len(lines)}\t{700000 if side == 'right' else 300000}\t{-(i % 5) * 1000}\t{i * 1000}\toptic\t\t{TYPES[i % 8]}\tacetylcholine\t{side}\tintrinsic")
    for t, ids in (("LPLC2", LPLC2), ("LC4", LC4), ("DNp01", DNP01), ("DNp04", DNP04)):
        for _ in ids:
            lines.append(f"{len(lines)}\t0\t0\t0\toptic\t\t{t}\tacetylcholine\tright\tafferent")
    p = tmp_path / "a.tsv"
    p.write_text("\n".join(lines) + "\n")
    return load_atlas(p)


def write_columns(path, atlas, skip=()):
    """A Codex-style column file for the atlas' T4/T5 (root ids = atlas root ids), minus the `skip` rows."""
    n = atlas.neurons
    rows = []
    for side in ("right", "left"):
        idx = [r for r in range(len(n)) if n["side"].iloc[r] == side and n["cell_type"].iloc[r] in TYPES]
        for r, (p, q) in zip(idx, LATTICE):
            if r not in skip:
                rows.append(f"{n['root_id'].iloc[r]},{side},{n['cell_type'].iloc[r]},{r},{(q - p) // 2},{p + q},{p},{q}")
    with gzip.open(path, "wt") as f:
        f.write("root_id,hemisphere,type,column_id,x,y,p,q\n" + "\n".join(rows) + "\n")
    return path


def test_angles_follow_the_measured_lens_map_at_known_columns():
    h, v = retina.column_angles(np.array([-4.0, 5.0]), np.array([8.0, -10.0]))
    assert h == pytest.approx([-0.807, 1.316], abs=0.05)  # radians, posterior +
    assert v == pytest.approx([0.162, -0.335], abs=0.05)  # up +


def test_frontal_columns_are_front_and_dorsal_columns_up_in_both_signs():
    h, v = retina.column_angles(np.array([-8.0, 8.0, 5.0, -5.0]), np.array([8.0, -8.0, 5.0, -5.0]))
    assert h[0] < 0 < h[1]  # x = (q - p) / 2: positive is frontward, so h (posterior +) is negative
    assert v[2] > 0 > v[3]  # y = p + q: positive is dorsal


def test_lattice_spacing_is_about_five_degrees_at_the_centre():
    h, v = retina.column_angles(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    step = np.degrees(np.hypot(h[1] - h[0], v[1] - v[0]))
    assert 3.0 < step < 7.0


def test_exact_eye_places_cells_by_column_and_normalises_to_unit_radius(atlas, tmp_path):
    path = write_columns(tmp_path / "c.csv.gz", atlas)
    emap, types, info = retina.exact_motion_eye(atlas, "right", path)
    assert len(emap) == len(types) == len(LATTICE) and info["undriven"] == 0
    assert np.linalg.norm(emap.xy, axis=1).max() == pytest.approx(1.0)
    by_row = dict(zip(emap.rows, emap.xy))
    first = int(emap.rows[0])
    assert by_row[first][0] < 0  # (-4, 8) is frontward


def test_cells_missing_from_the_column_file_stay_undriven_and_are_counted(atlas, tmp_path):
    n = atlas.neurons
    skip = {r for r in range(len(n)) if n["side"].iloc[r] == "right" and n["cell_type"].iloc[r] in TYPES}
    skip = set(sorted(skip)[:3])
    path = write_columns(tmp_path / "c.csv.gz", atlas, skip=skip)
    emap, types, info = retina.exact_motion_eye(atlas, "right", path)
    assert len(emap) == len(LATTICE) - 3 and info["undriven"] == 3 and not set(emap.rows) & skip


def test_a_missing_file_or_no_matching_cell_gives_none(atlas, tmp_path):
    assert retina.exact_motion_eye(atlas, "right", tmp_path / "nope.csv.gz") is None
    junk = tmp_path / "junk.csv.gz"
    with gzip.open(junk, "wt") as f:
        f.write("root_id,hemisphere,type,column_id,x,y,p,q\n1,right,T4a,1,0,0,0,0\n")
    assert retina.exact_motion_eye(atlas, "right", junk) is None


# --- looming_escape uses it when the file is there, falls back to the PCA map when it is not -----------------

@pytest.fixture
def sims(atlas):
    edges = [(r, d, 1000) for r in range(0, 20) for d in LPLC2 + LC4]
    edges += [(d, t, 1000) for d in LPLC2 + LC4 for t in DNP01]
    pre, post, w = (np.array(c) for c in zip(*edges))
    return SimulationService.from_simulation(Simulation.from_arrays(pre, post, w, n_neurons=len(atlas.neurons)))


def run(atlas, sims, columns_path, **kw):
    opts = {"input": "motion", "rate_hz": 300, "duration_ms": 300, "trials": 2, "steps": 6, "polarity": "both",
            "columns_path": columns_path, **kw}
    return looming.looming_escape(atlas, sims, **opts)


def test_motion_uses_the_exact_map_and_says_so_with_the_undriven_count(atlas, sims, tmp_path):
    n = atlas.neurons
    first = min(r for r in range(len(n)) if n["side"].iloc[r] == "right" and n["cell_type"].iloc[r] in TYPES)
    out = run(atlas, sims, write_columns(tmp_path / "c.csv.gz", atlas, skip={first}))
    assert "exact retinotopy" in out["note"] and "1 T4/T5 cells" in out["note"]
    assert out["input"] == "motion"


def test_motion_falls_back_to_the_pca_map_with_a_clear_note(atlas, sims, tmp_path):
    out = run(atlas, sims, tmp_path / "missing.csv.gz")
    assert "PCA" in out["note"] and "column file" in out["note"] and "exact retinotopy" not in out["note"]


def test_equal_total_drive_holds_with_the_exact_map(atlas, sims, tmp_path, monkeypatch):
    seen = []
    real = sims.run_schedule

    def spy(schedule, *a, **k):
        seen.append(schedule)
        return real(schedule, *a, **k)

    monkeypatch.setattr(sims, "run_schedule", spy)
    run(atlas, sims, write_columns(tmp_path / "c.csv.gz", atlas))
    drives = [eye.schedule_drive(s) for s in seen]
    assert len(drives) == 4 and drives[0] > 0 and drives == pytest.approx([drives[0]] * 4, rel=1e-6)
