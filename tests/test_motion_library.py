"""web/public/fly-motion-library.json: the committed motion library built by scripts/build_motion_library.py.

Checked as data (no flygym): schema, grid, phase alignment, loop seams, speed / yaw-rate monotonicity, size."""

import json
from pathlib import Path

import numpy as np
import pytest

from flybrain.body_kinematic import LIBRARY_PATH, load_library

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "web" / "public" / "fly-motion-library.json"


@pytest.fixture(scope="module")
def raw():
    return json.loads(ASSET.read_text())


@pytest.fixture(scope="module")
def lib():
    return load_library()


def test_default_path_is_the_committed_asset():
    assert LIBRARY_PATH.resolve() == ASSET.resolve() and ASSET.exists()


def test_schema_provenance_and_size(raw, lib):
    assert raw["format"] == "fly-motion-library/1"
    assert raw["sample_rate_hz"] == 200 and raw["cycle_hz"] == 12.0 and raw["quat_scale"] == 10000
    assert len(raw["bodies"]) == 48 == len(set(raw["bodies"])) and len(raw["legs"]) == 6
    assert raw["encoding"]["dtype"] == "<i2" and raw["encoding"]["shape"] == ["phase", "body", 7]
    p = raw["provenance"]
    assert p["generator"] == "scripts/build_motion_library.py" and p["flygym"] and p["mujoco"] and p["physics_dt_s"] > 0
    assert "lf) swing onset" in raw["phase_alignment"].lower()
    assert ASSET.stat().st_size < 700_000
    assert lib.phase_samples == raw["phase_samples"] >= 16


def test_grid_covers_pace_and_turn_plus_spin_and_stand(raw, lib):
    walk = [c for c in lib.clips if c.kind == "walk"]
    assert len(lib.pace_grid) >= 4 and len(lib.turn_grid) >= 5
    assert lib.pace_grid[0] == 0.0 and lib.pace_grid[-1] == 1.0 and lib.turn_grid[0] == -1.0 and lib.turn_grid[-1] == 1.0
    assert {(c.pace, c.turn) for c in walk} == {(p, t) for p in lib.pace_grid for t in lib.turn_grid}
    assert {c.kind for c in lib.clips} <= {"walk", "spin"}
    assert sorted(c.turn for c in lib.clips if c.kind == "spin") in ([], [-1.0, 1.0])
    assert lib.stand.pos.shape == (48, 3) and lib.stand.quat.shape == (48, 4) and lib.stand.contact.shape == (6,)
    assert raw["escape"]["pose"] == "stand" and raw["escape"]["wings_raised"] is True
    for c in lib.clips:
        assert c.pos.shape == (lib.phase_samples, 48, 3) and c.quat.shape == (lib.phase_samples, 48, 4)
        assert c.contact.shape == (lib.phase_samples, 6)
        assert np.allclose(np.linalg.norm(c.quat, axis=-1), 1.0, atol=2e-3)
        assert np.isfinite(c.pos).all() and np.abs(c.pos).max() < 5000  # um, leg bodies around the thorax


def test_every_quaternion_sits_in_the_hemisphere_of_the_neutral_pose(lib):
    for c in lib.clips:
        assert (np.sum(c.quat * lib.neutral_quat[None], axis=-1) > 0).all()
    assert (np.sum(lib.stand.quat * lib.neutral_quat, axis=-1) > 0).all()


def test_speed_grows_with_pace_and_yaw_rate_follows_the_turn_sign(lib):
    straight = sorted((c for c in lib.clips if c.kind == "walk" and c.turn == 0.0), key=lambda c: c.pace)
    speeds = [c.speed_mm_s for c in straight]
    assert all(b > a for a, b in zip(speeds, speeds[1:], strict=False)) and speeds[0] > 3.0
    assert all(abs(c.yaw_rate_deg_s) < 20 for c in straight)  # straight walking barely rotates
    for c in lib.clips:
        if c.kind == "walk" and c.turn > 0:
            assert c.yaw_rate_deg_s > 5.0  # positive turn = left = positive yaw (counter-clockwise from above)
        if c.kind == "walk" and c.turn < 0:
            assert c.yaw_rate_deg_s < -5.0
    for p in lib.pace_grid:
        rates = [c.yaw_rate_deg_s for c in sorted((c for c in lib.clips if c.kind == "walk" and c.pace == p),
                                                   key=lambda c: c.turn)]
        assert all(b > a for a, b in zip(rates, rates[1:], strict=False))  # monotone in the turn command


def test_loops_have_no_seam_the_wrap_step_is_like_any_other_step(lib):
    for c in lib.clips:
        steps = np.abs(np.diff(np.concatenate([c.pos, c.pos[:1]]), axis=0)).max(axis=(1, 2))  # incl. last -> first
        interior = steps[:-1].max()
        assert steps[-1] <= 1.5 * interior, (c.id, steps[-1], interior)
        assert c.seam["wrap_over_max_step"] == pytest.approx(steps[-1] / interior, rel=0.05, abs=0.05)
        assert c.seam["repeat_residual_um"] < 150  # cycles of the recording agree


def test_clips_are_phase_aligned_at_the_lf_swing_onset(lib):
    """Against the fastest straight clip, every walking clip is best matched with NO phase shift (+-1 sample of 25)."""
    ref = lib.clip_at(lib.pace_grid[-1], 0.0)
    P = lib.phase_samples
    for c in (c for c in lib.clips if c.kind == "walk"):
        cost = [np.abs(np.roll(c.pos, -s, axis=0) - ref.pos).mean() for s in range(P)]
        best = int(np.argmin(cost))
        assert min(best, P - best) <= 1, (c.id, best)


def test_a_walk_cycle_actually_moves_the_legs_and_standing_is_still(lib):
    c = lib.clip_at(1.0, 0.0)
    assert np.ptp(c.pos[:, lib.bodies.index("lf_tarsus5")], axis=0).max() > 300  # um of stepping
    assert np.abs(c.pos).max() > 0 and np.abs(lib.stand.pos - lib.neutral_pos).max() < 800  # flygym's own default pose is ~0.5 mm from the spec neutral


def test_weights_pick_the_nearest_clips_normalised_and_exact_on_a_grid_point(lib):
    p, t = lib.pace_grid[2], lib.turn_grid[3]
    w = lib.weights(p, t)
    assert w == {lib.clip_at(p, t).id: 1.0}
    pm, tm = (lib.pace_grid[1] + lib.pace_grid[2]) / 2, (lib.turn_grid[2] + lib.turn_grid[3]) / 2
    w = lib.weights(pm, tm)
    assert len(w) == 4 and sum(w.values()) == pytest.approx(1.0) and all(v == pytest.approx(0.25) for v in w.values())
    assert {(lib.clip_by_id(i).pace, lib.clip_by_id(i).turn) for i in w} == {
        (a, b) for a in lib.pace_grid[1:3] for b in lib.turn_grid[2:4]}
    w = lib.weights(lib.pace_grid[1], tm)  # on a pace line, between two turns: two clips
    assert len(w) == 2 and sum(w.values()) == pytest.approx(1.0)


def test_weights_clamp_outside_the_grid_and_reject_garbage(lib):
    assert lib.weights(5.0, -9.0) == lib.weights(lib.pace_grid[-1], lib.turn_grid[0])
    assert lib.weights(-3.0, 0.0) == lib.weights(lib.pace_grid[0], 0.0)
    with pytest.raises(ValueError):
        lib.weights(float("nan"), 0.0)


def test_loading_a_wrong_format_fails_loudly(tmp_path):
    bad = tmp_path / "lib.json"
    bad.write_text(json.dumps({"format": "other/1"}))
    with pytest.raises(ValueError, match="fly-motion-library/1"):
        load_library(bad)
