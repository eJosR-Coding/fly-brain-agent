"""KinematicBody: the recorded physics gait, blended live (pure numpy; the physics comparison is gated on flygym)."""

import importlib.util

import numpy as np
import pytest

from flybrain import body_drive
from flybrain.body_kinematic import KinematicBody, load_library
from flybrain.live import LiveSession
from flybrain.simulation import ChunkSpikes

HAS_FLYGYM = importlib.util.find_spec("flygym") is not None
phys = pytest.mark.skipif(not HAS_FLYGYM, reason="flygym is an optional extra: uv sync --extra body")
LIB = load_library()


def body(**kw):
    return KinematicBody(LIB, **kw)


def run(b, n_chunks, pace=0.8, turn=0.0, escape=False, ms=50.0):
    outs = []
    for _ in range(n_chunks):
        outs.append(b.advance(ms, body_drive.walking_drive(pace, turn), escape, pace=pace, turn=turn))
    return outs


def cat(outs, key):
    return np.concatenate([np.asarray(o[key]) for o in outs])


def test_chunk_has_the_bodystream_shape_and_the_header_describes_the_clip_format():
    b = body()
    out = b.advance(50.0, [1.0, 1.0], False, pace=1.0, turn=0.0)
    assert out["n_frames"] == 10 and out["frames"].shape == (10, 48, 7) and out["frames"].dtype.kind == "i"
    assert out["contact"].shape == (10, 6) and out["escape"].shape == (10,) and out["drive"].shape == (10, 2)
    assert out["thorax"]["pos_um"].shape == (10, 3) and out["thorax"]["yaw_deg"].shape == (10,)
    h = b.header
    assert h["format"] == "fly-walk/1" and len(h["bodies"]) == 48 and h["sample_rate_hz"] == 200
    assert h["body_mode"] == "kinematic" and "recorded physics gait" in b.caption
    assert b.mode == "kinematic" and b.t_ms == pytest.approx(50.0)


def test_chunk_must_be_whole_frames_and_drive_finite():
    b = body()
    with pytest.raises(ValueError):
        b.advance(7.0, [1, 1], False)
    with pytest.raises(ValueError):
        b.advance(50.0, [1, float("nan")], False)


def test_chunking_does_not_change_the_motion_and_phase_is_continuous():
    a, c = body(), body()
    fa = cat(run(a, 4), "frames")
    fc = c.advance(200.0, body_drive.walking_drive(0.8, 0.0), False, pace=0.8, turn=0.0)["frames"]
    assert np.array_equal(fa, fc)
    steps = np.abs(np.diff(fa.astype(float)[:, :, :3], axis=0)).max()
    assert steps <= 450  # um between 5 ms frames: no phase jump at chunk edges (a swing is ~100 um/frame)
    assert a.phase == pytest.approx((0.06 * 40) % 1.0)  # 12 Hz at 200 Hz


def test_a_grid_point_plays_exactly_that_clip_after_the_start_ramp():
    c = LIB.clip_at(LIB.pace_grid[-1], 0.0)
    frames = cat(run(body(), 4, pace=c.pace, turn=0.0), "frames")
    k = 30  # past the 100 ms start ramp; at even k the phase (0.06 k x 25 = 1.5 k) lands on a stored phase sample
    got = frames[k, :, :3].astype(float)
    assert np.abs(got - np.round(c.pos[(3 * k // 2) % LIB.phase_samples])).max() <= 1.0
    q = frames[k, :, 3:] / 10000.0
    assert np.abs(q - c.quat[(3 * k // 2) % LIB.phase_samples]).max() < 2e-3


def test_blend_is_continuous_in_the_command():
    a, c = body(), body()
    fa = run(a, 6, pace=0.5, turn=0.2)
    fc = run(c, 6, pace=0.5 + 1e-4, turn=0.2 - 1e-4)
    assert np.abs(cat(fa, "frames") - cat(fc, "frames")).max() <= 2


def test_stop_returns_smoothly_to_the_standing_pose_within_100_ms():
    b = body()
    run(b, 6)
    out = b.advance(200.0, [0.0, 0.0], False, pace=0.0, turn=0.0)
    frames = out["frames"].astype(float)
    assert np.abs(np.diff(frames[:, :, :3], axis=0)).max() < 400  # ramp, no jump
    stand = np.round(LIB.stand.pos)
    assert np.abs(frames[21:, :, :3] - stand).max() <= 1.0  # 100 ms = 20 frames
    assert np.abs(frames[5, :, :3] - stand).max() > 5  # ... and not instantly
    assert (out["thorax"]["yaw_deg"][21:] == out["thorax"]["yaw_deg"][21]).all()  # no root motion once still
    assert (np.diff(out["thorax"]["pos_um"][21:, 0]) == 0).all()
    assert out["contact"][-1].all() == bool(LIB.stand.contact.all())


def test_zero_drive_from_the_start_just_stands():
    out = body().advance(50.0, [0.0, 0.0], False, pace=0.0, turn=0.0)
    assert np.abs(out["frames"][:, :, :3] - np.round(LIB.stand.pos)).max() <= 1.0


def test_straight_root_motion_integrates_the_measured_speed():
    c = LIB.clip_at(LIB.pace_grid[-1], 0.0)
    b = body()
    outs = run(b, 20, pace=c.pace, turn=0.0)  # 1 s
    pos = np.concatenate([o["thorax"]["pos_um"] for o in outs])
    yaw = np.concatenate([o["thorax"]["yaw_deg"] for o in outs])
    walked_mm = pos[-1, 0] / 1000.0
    ramp_loss = 0.05 * c.speed_mm_s  # the 100 ms start ramp walks at ~half speed for 100 ms
    assert walked_mm == pytest.approx(c.speed_mm_s - ramp_loss, rel=0.03)
    assert abs(yaw[-1]) == pytest.approx(abs(c.yaw_rate_deg_s) * 0.95, abs=1.0)  # the clip's own small drift only


def test_positive_turn_curves_left_negative_right_and_yaw_rate_matches_the_blend():
    for sign in (1, -1):
        b = body()
        outs = run(b, 20, pace=0.5, turn=0.5 * sign)
        pos = np.concatenate([o["thorax"]["pos_um"] for o in outs])
        yaw = np.concatenate([o["thorax"]["yaw_deg"] for o in outs])
        assert np.sign(yaw[-1]) == sign and np.sign(pos[-1, 1]) == sign  # +yaw = left = +y
        w = LIB.weights(0.5, 0.5 * sign)
        rate = sum(wt * LIB.clip_by_id(i).yaw_rate_deg_s for i, wt in w.items())
        assert yaw[-1] == pytest.approx(rate * 0.95, rel=0.05)


def test_the_drive_argument_alone_maps_back_to_pace_and_turn():
    a, c = body(), body()
    fa = a.advance(100.0, body_drive.walking_drive(0.7, 0.4), False)["frames"]
    fc = c.advance(100.0, body_drive.walking_drive(0.7, 0.4), False, pace=0.7, turn=0.4)["frames"]
    assert np.abs(fa.astype(int) - fc).max() <= 2


def test_escape_stands_raises_the_flag_and_stops_the_root():
    b = body()
    run(b, 4)
    out = b.advance(200.0, [1.0, 1.0], True, pace=1.0, turn=0.0)
    assert out["escape"].all() and (out["drive"] == 0).all()
    assert np.abs(out["frames"][21:, :, :3] - np.round(LIB.stand.pos)).max() <= 1.0
    assert (np.diff(out["thorax"]["pos_um"][21:, 0]) == 0).all()
    after = b.advance(50.0, [1.0, 1.0], False, pace=1.0, turn=0.0)
    assert not after["escape"].any()


def test_a_world_boundary_stops_the_fly_at_the_wall_and_says_so():
    b = body(bounds=(-5.0, 5.0, -5.0, 5.0))  # mm
    outs = run(b, 60, pace=1.0, turn=0.0)  # 3 s at ~20 mm/s would reach 60 mm
    pos = np.concatenate([o["thorax"]["pos_um"] for o in outs])
    assert pos[:, 0].max() <= 5000 and pos[-1, 0] == pytest.approx(5000, abs=1)
    assert outs[-1]["world"]["blocked"] is True and outs[0]["world"]["blocked"] is False
    assert "world rule" in outs[-1]["world"]["note"].lower()
    free = run(body(), 60, pace=1.0, turn=0.0)
    assert np.concatenate([o["thorax"]["pos_um"] for o in free])[-1, 0] > 5000 and free[-1]["world"]["blocked"] is False
    assert body().bounds is None


def test_the_body_is_deterministic():
    a, c = body(), body()
    assert np.array_equal(cat(run(a, 5, 0.6, -0.3), "frames"), cat(run(c, 5, 0.6, -0.3), "frames"))


class _Brain:
    backend = "cpu"

    def set_stimulus(self, rows, rate_hz):
        pass

    def advance(self, ms):
        rows = np.array([0, 1], dtype=np.int64)
        return ChunkSpikes(rows, np.array([2, 2], dtype=np.int32), ms)


def test_live_session_accepts_the_kinematic_body_and_reports_the_mode():
    groups = {"dng100": np.array([0, 1]), "dna01_left": np.array([2]), "dna01_right": np.array([3]),
              "dna02_left": np.array([4]), "dna02_right": np.array([5]), "dnp01": np.array([6, 7])}
    s = LiveSession(_Brain(), body(), groups)
    started = s.start_message()
    assert started["body_mode"] == "kinematic" and "recorded physics gait" in started["body_caption"]
    msg = None
    for _ in range(8):
        msg = s.step()
    assert msg["body_mode"] == "kinematic" and "blended live" in msg["body_caption"]
    assert np.asarray(msg["frames"]).shape == (10, 48, 7) and msg["thorax"]["pos_um"][-1][0] > 0
    m2, blob = s.step_binary()
    assert m2["frames_binary"]["shape"] == [10, 48, 7] and len(blob) == 10 * 48 * 7 * 2


@phys
def test_kinematic_speed_and_yaw_match_the_physics_run_it_was_recorded_from():
    from flybrain.body_runner import run_body

    pace, turn = 0.67, 0.5
    drive = body_drive.walking_drive(pace, turn)
    phys_clip = run_body(np.tile(drive, (60, 1)), 50.0, 3000.0, seed=0)  # 3 s, constant drive
    w = phys_clip["thorax_world"]
    i0, i1 = 200, 599  # t = 1.0 .. 3.0 s, after the physics start-up

    def measure(pos_um, yaw_deg):
        """(forward speed mm/s along the heading, yaw rate deg/s) over frames i0..i1, the library's own definition."""
        pos = np.asarray(pos_um, float)[:, :2] / 1000.0
        yaw = np.radians(np.asarray(yaw_deg, float))
        mid = 0.5 * (yaw[i0 + 1:i1 + 1] + yaw[i0:i1])
        vel = np.diff(pos[i0:i1 + 1], axis=0) * 200.0
        fwd = np.mean(vel[:, 0] * np.cos(mid) + vel[:, 1] * np.sin(mid))
        return float(fwd), float((yaw_deg[i1] - yaw_deg[i0]) / ((i1 - i0) / 200))

    phys_speed, phys_rate = measure(w["pos_um"], w["yaw_deg"])
    outs = run(body(), 60, pace=pace, turn=turn)
    kin_speed, kin_rate = measure(np.concatenate([o["thorax"]["pos_um"] for o in outs]),
                                  np.concatenate([o["thorax"]["yaw_deg"] for o in outs]))
    assert kin_speed == pytest.approx(phys_speed, rel=0.15)
    assert kin_rate == pytest.approx(phys_rate, rel=0.20)
