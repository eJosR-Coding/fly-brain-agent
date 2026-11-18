"""B2: the body runner. Pure parts always; the physics integration only when flygym is importable."""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from flybrain import body_drive as bd
from flybrain import body_runner as br

HAS_FLYGYM = all(importlib.util.find_spec(m) is not None for m in ("flygym", "flygym_demo", "mujoco"))


# ---- pure ----

def test_schedule_interpolates_between_bin_centres_and_holds_the_ends():
    s, esc = br.drive_schedule([[0.0, 0.0], [1.0, 0.5]], None, bin_ms=100.0, n_steps=201, dt_s=0.001)
    assert s.shape == (201, 2) and not esc.any()
    assert s[0].tolist() == [0.0, 0.0]  # before the first centre (50 ms): held
    assert s[100] == pytest.approx([0.5, 0.25])  # t = 100 ms: halfway between the centres at 50 and 150 ms
    assert s[200] == pytest.approx([1.0, 0.5])  # after the last centre: held


def test_schedule_escape_is_held_per_bin_and_forces_zero_drive():
    s, esc = br.drive_schedule([[1, 1], [1, 1], [1, 1]], [False, True, False], bin_ms=100.0, n_steps=300, dt_s=0.001)
    assert esc[:100].sum() == 0 and esc[100:200].all() and esc[200:].sum() == 0
    assert (s[esc] == 0).all() and (s[~esc] > 0).all()


def test_schedule_without_bins_is_a_stopped_fly():
    s, esc = br.drive_schedule([], None, bin_ms=50.0, n_steps=5, dt_s=0.001)
    assert (s == 0).all() and not esc.any()


def test_yaw_sign_positive_is_counter_clockwise_and_unit_quaternion_free():
    half = np.radians(30) / 2  # +30 degrees about z (up): the x axis turns towards +y (left)
    q = np.array([[np.cos(half), 0, 0, np.sin(half)], [1, 0, 0, 0]])
    assert br.yaw_deg(q) == pytest.approx([30.0, 0.0])


def test_continuous_keeps_one_hemisphere():
    q = np.array([[[1, 0, 0, 0]], [[-1, 0, 0, 0.01]]], dtype=float)
    out = br.continuous(q)
    assert (np.sum(out[1] * out[0], axis=-1) > 0).all()


def test_bodies_layout_matches_the_committed_walking_asset():
    asset = json.loads((Path(__file__).resolve().parents[1] / "web/public/fly-walk.json").read_text())
    assert asset["bodies"] == br.BODIES and asset["quat_scale"] == br.QUAT_SCALE and asset["sample_rate_hz"] == br.SAMPLE_HZ


def test_missing_flygym_raises_a_clear_error_with_the_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "flygym", None)  # makes `import flygym` raise ImportError
    with pytest.raises(br.BodyUnavailable, match=r"uv sync --extra body"):
        br.run_body([[1, 1]], 50.0, 50.0)
    assert issubclass(br.BodyUnavailable, RuntimeError)


# ---- physics (gated) ----

pytestmark_phys = pytest.mark.skipif(not HAS_FLYGYM, reason="flygym is an optional extra: uv sync --extra body")
TIMINGS: dict = {}


def run_const(drive, ms=300.0, escape=False):
    t0 = time.perf_counter()
    clip = br.run_body([list(drive)] * 3, ms / 3, ms, seed=0, escape=[escape] * 3)
    TIMINGS[tuple(drive) + (escape,)] = (time.perf_counter() - t0) / (ms / 1000.0)
    return clip


def leg_motion(clip):
    f = np.array(clip["frames"], dtype=float)
    i = [clip["bodies"].index(f"{leg}_tarsus5") for leg in clip["legs"]]
    return f[len(f) // 2 :, i, :3].std(axis=0).max()  # um, second half of the clip


@pytestmark_phys
def test_clip_schema_and_viewer_compatibility():
    clip = run_const((1.0, 1.0), ms=100.0)
    n = clip["n_frames"]
    assert clip["format"] == "fly-walk/1" and n == 20 and clip["loop"] is False
    assert np.array(clip["frames"]).shape == (n, 48, 7)
    assert np.array(clip["adhesion"]).shape == (n, 6) and len(clip["escape"]) == n == len(clip["drive"])
    tw = clip["thorax_world"]
    assert np.array(tw["pos_um"]).shape == (n, 3) and len(tw["yaw_deg"]) == n
    assert tw["pos_um"][0][:2] == [0, 0] and tw["yaw_deg"][0] == 0
    json.dumps(clip)
    q = np.array(clip["frames"])[:, :, 3:] / clip["quat_scale"]
    assert np.abs(np.linalg.norm(q, axis=-1) - 1).max() < 1e-3


@pytestmark_phys
def test_forward_drive_moves_the_thorax_forward_without_turning():
    clip = run_const((1.0, 1.0))
    pos, yaw = np.array(clip["thorax_world"]["pos_um"]), clip["thorax_world"]["yaw_deg"]
    assert pos[-1][0] > 1000  # more than 1 mm forward in 0.3 s
    # equal drive does not turn on purpose; the controller leaves a small start-up heading offset (about +15 deg seen)
    assert abs(yaw[-1]) < 25 and abs(pos[-1][1]) < pos[-1][0]


@pytestmark_phys
def test_turn_drive_changes_yaw_with_the_documented_sign():
    straight = run_const((1.0, 1.0))["thorax_world"]["yaw_deg"][-1]
    left = run_const((0.4, 1.0))["thorax_world"]["yaw_deg"][-1]  # lower LEFT drive: turn left, positive yaw
    right = run_const((1.0, 0.4))["thorax_world"]["yaw_deg"][-1]
    assert left > straight + 10 and right < straight - 10, (left, straight, right)


@pytestmark_phys
def test_escape_stops_leg_motion():
    walking, stopped = leg_motion(run_const((1.0, 1.0))), leg_motion(run_const((1.0, 1.0), escape=True))
    assert walking > 300 and stopped < 0.1 * walking, (walking, stopped)
    clip = run_const((1.0, 1.0), escape=True)
    assert all(clip["escape"]) and np.array(clip["drive"]).max() == 0


@pytestmark_phys
def test_runner_takes_a_body_drive_object():
    d = bd.drive_from_readouts([{"name": "DNg100", "hz": [60, 60]}], bin_ms=50.0)
    clip = br.run_body(d, 50.0, 100.0)
    assert clip["n_frames"] == 20 and clip["thorax_world"]["pos_um"][-1][0] > 0


# ---- physics timestep (G2): the default step reproduces the dt = 1e-4 s reference walking ----

TRIPOD_A, TRIPOD_B = ("LF", "RM", "LH"), ("RF", "LM", "RH")


def tripod_antiphase(clip):
    """Mean correlation of the stance (adhesion) signals within a tripod, and across the two tripods."""
    a = np.array(clip["adhesion"], dtype=float)[20:]  # skip the first 0.1 s
    legs = [str(leg).upper() for leg in clip["legs"]]
    c = lambda x, y: np.corrcoef(a[:, legs.index(x)], a[:, legs.index(y)])[0, 1]  # noqa: E731
    within = np.mean([c(g[i], g[j]) for g in (TRIPOD_A, TRIPOD_B) for i, j in ((0, 1), (0, 2), (1, 2))])
    across = np.mean([c(x, y) for x in TRIPOD_A for y in TRIPOD_B])
    return within, across


@pytestmark_phys
def test_default_physics_step_matches_the_reference_step():
    assert br.DEFAULT_DT_S > 1e-4  # it is the measured faster step, not the reference itself
    ref_dt = 1e-4
    out = {}
    for name, drive in (("straight", (1.0, 1.0)), ("turn", (0.4, 1.2))):
        out[name] = [br.run_body([list(drive)] * 2, 250.0, 500.0, seed=0, escape=[False] * 2, dt_s=dt)
                     for dt in (ref_dt, br.DEFAULT_DT_S)]
    ref, new = out["straight"]
    speed = lambda c: np.hypot(*np.array(c["thorax_world"]["pos_um"])[-1][:2]) / 1000.0 / c["duration_s"]  # noqa: E731
    assert speed(new) == pytest.approx(speed(ref), rel=0.10)
    ref, new = out["turn"]
    yaw = lambda c: c["thorax_world"]["yaw_deg"][-1] / c["duration_s"]  # noqa: E731
    assert yaw(ref) > 50 and yaw(new) == pytest.approx(yaw(ref), rel=0.15)
    for clip in (out["straight"][1], out["turn"][1]):
        within, across = tripod_antiphase(clip)
        assert within > 0.8 and across < -0.8, (within, across)
        assert clip["provenance"]["physics_dt_s"] == pytest.approx(br.DEFAULT_DT_S)
        assert np.isfinite(np.array(clip["frames"], dtype=float)).all()


def test_sample_stride_needs_a_step_that_divides_the_frame_period():
    assert [br.sample_stride(dt) for dt in (1e-4, 2e-4, 5e-4)] == [50, 25, 10]
    with pytest.raises(ValueError, match="divide"):
        br.sample_stride(3e-4)
