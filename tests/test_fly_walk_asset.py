"""web/public/fly-walk.json: the physics walking clip exported by scripts/export_flygym_walk.py (flygym 2.1.1).

The clip is checked as data (shape, loop, stepping frequency, tripod coordination); the assembled-mesh gate (our body
origins against MuJoCo's neutral-pose origins) runs when FLYBRAIN_FLYGYM_WHEEL points at the verified flygym wheel.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
WALK = ROOT / "web" / "public" / "fly-walk.json"
BODY = ROOT / "web" / "public" / "fly-body.json"
WHEEL_ENV = "FLYBRAIN_FLYGYM_WHEEL"

TRIPOD_A = ("lf", "rm", "lh")
TRIPOD_B = ("rf", "lm", "rh")


@pytest.fixture(scope="module")
def clip():
    return json.loads(WALK.read_text())


def frames(clip):
    return np.array(clip["frames"], dtype=float)  # (frame, body, [x y z qw qx qy qz])


def test_shape_provenance_and_size(clip):
    assert clip["format"] == "fly-walk/1"
    assert len(clip["bodies"]) == 48 == len(set(clip["bodies"]))
    parts = {p["name"] for p in json.loads(BODY.read_text())["parts"]}
    assert set(clip["bodies"]) <= parts
    assert clip["sample_rate_hz"] == 200 and clip["n_frames"] == 200 and clip["duration_s"] == 1.0
    assert clip["cycle_hz"] * clip["duration_s"] == clip["cycles"] == 12
    f = frames(clip)
    assert f.shape == (200, 48, 7)
    assert np.array(clip["adhesion"]).shape == (200, 6) == np.array(clip["contact"]).shape
    prov = clip["provenance"]
    assert prov["flygym"] == "2.1.1" and prov["mujoco"].startswith("3.")
    assert prov["drive"] == [1.0, 1.0] and prov["generator"] == "scripts/export_flygym_walk.py"
    assert WALK.stat().st_size < 500_000


def test_quaternions_are_unit_and_continuous_so_the_viewer_never_spins_the_long_way(clip):
    q = frames(clip)[:, :, 3:] / clip["quat_scale"]
    assert np.abs(np.linalg.norm(q, axis=-1) - 1).max() < 1e-3
    assert (np.sum(q[1:] * q[:-1], axis=-1) > 0).all()
    assert (np.sum(q[0] * (np.array(clip["neutral"]["quat"]) / clip["quat_scale"]), axis=-1) > 0).all()


def test_the_loop_closes(clip):
    seam = clip["provenance"]["loop_seam"]
    assert seam["max_position_um"] < 20 and seam["max_rotation_deg"] < 1.5
    assert seam["max_position_um"] < seam["max_frame_to_frame_um"] / 20  # far below an ordinary step of the clip


def test_stepping_frequency_is_12_hz_in_every_leg(clip):
    f = frames(clip)
    for leg in clip["legs"]:
        x = f[:, clip["bodies"].index(f"{leg}_tarsus5"), 0]
        spectrum = np.abs(np.fft.rfft(x - x.mean()))
        assert np.argmax(spectrum) * clip["sample_rate_hz"] / len(x) == pytest.approx(12.0)
    assert 5 <= clip["provenance"]["thorax_speed_mm_s"] <= 30  # real flies walk about 5-30 mm/s


def test_tripod_coordination_two_tripods_in_antiphase(clip):
    legs = clip["legs"]
    f = frames(clip)
    stance = np.array(clip["adhesion"], dtype=float)  # the controller's stance command (adhesion on)
    # swing = the foot moves forward relative to the thorax (circular central difference of the tarsus5 x)
    swing = []
    for leg in legs:
        x = f[:, clip["bodies"].index(f"{leg}_tarsus5"), 0]
        swing.append(((np.roll(x, -1) - np.roll(x, 1)) > 0).astype(float))
    swing = np.array(swing)
    for series, within, between in ((stance.T, 0.75, -0.75), (swing, 0.35, -0.35)):
        corr = np.corrcoef(series)
        idx = {leg: i for i, leg in enumerate(legs)}
        same = [corr[idx[a], idx[b]] for grp in (TRIPOD_A, TRIPOD_B) for a in grp for b in grp if a < b]
        cross = [corr[idx[a], idx[b]] for a in TRIPOD_A for b in TRIPOD_B]
        assert min(same) > within, same
        assert max(cross) < between, cross


@pytest.mark.skipif(
    not (os.environ.get(WHEEL_ENV) and Path(os.environ.get(WHEEL_ENV, "")).exists()),
    reason=f"set {WHEEL_ENV} to the verified flygym wheel",
)
def test_gate_our_assembled_body_origins_match_mujocos_neutral_pose_origins(clip):
    from flybrain.fly_body_build import body_origins

    origins = body_origins(Path(os.environ[WHEEL_ENV]))
    thorax = origins["c_thorax"]
    assert np.allclose(clip["thorax_origin_um"], thorax, atol=1.0)
    errs = []
    for body, mujoco in zip(clip["bodies"], clip["neutral"]["pos_um"]):
        ours = origins["l" + body[1:]] - thorax  # right side = the left mirrored about y
        if body[0] == "r":
            ours = ours * np.array([1.0, -1.0, 1.0])
        errs.append(np.linalg.norm(ours - np.array(mujoco, dtype=float)))
    assert max(errs) < 2.0  # um: the integer rounding of the exported positions is the only difference
    assert np.median(errs) < 1.0
