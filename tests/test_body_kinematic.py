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


def test_surface_walking_carries_the_fly_up_the_wall_and_onto_the_ceiling():
    b = body(bounds=(-5.0, 5.0, -5.0, 5.0))  # mm: a 10 mm cube
    outs = run(b, 150, pace=1.0, turn=0.0)  # 7.5 s at ~20 mm/s
    trans = [(t["from"], t["to"]) for o in outs for t in o["world"]["transitions"]]
    assert trans[:2] == [("floor", "x_max"), ("x_max", "ceiling")]  # head-on: straight up the wall, then upside down
    pos = np.concatenate([o["thorax"]["pos_um"] for o in outs])
    assert pos[:, 0].max() <= 5000 and pos[:, 2].max() <= 10000 and pos[:, 2].min() >= 0  # the thorax never leaves the cube
    assert "world rule" in outs[-1]["world"]["note"].lower()
    from flybrain.body_surface import matrix_from_quat
    on_ceiling = [o for o in outs if o["world"]["surface"] == "ceiling"]
    assert on_ceiling and matrix_from_quat(on_ceiling[-1]["thorax"]["quat"][-1])[2, 2] < -0.9  # dorsal axis points down on the ceiling
    free = run(body(), 60, pace=1.0, turn=0.0)
    assert np.concatenate([o["thorax"]["pos_um"] for o in free])[-1, 0] > 5000 and free[-1]["world"]["surface"] == "floor"
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


def test_gait_keeps_its_native_rate_on_every_surface():
    walled = KinematicBody(bounds=(-1, 1, -50, 50))  # reaches a wall at once
    for _ in range(40):
        walled.advance(100, [1, 1], pace=0.67, turn=0.0)
    assert walled.cycles == pytest.approx(48.0)  # adhesive pads: the legs step at the native 12 Hz on the wall too


def _fly(bounds=None, away=180.0, kind="straight"):
    b = KinematicBody(bounds=bounds)
    run(b, 2, pace=0.0)
    assert b.takeoff(away, kind) and not b.takeoff(away)  # busy
    outs = [b.advance(50.0, [0.0, 0.0], False, pace=0.0, turn=0.0) for _ in range(220)]
    return b, outs


def test_takeoff_flies_a_real_clip_then_descends_lands_and_stands():
    b, outs = _fly()
    seq = [s for o in outs for s in o["state_seq"]]
    assert [s for i, s in enumerate(seq) if i == 0 or seq[i - 1] != s][:4] == ["takeoff", "airborne", "descending", "landed"]
    z = np.concatenate([o["thorax"]["pos_um"][:, 2] for o in outs])
    assert z.max() > z.min() + 2000 and outs[-1]["state"] == "standing" and not b.busy
    assert z[-1] == pytest.approx(z[0], abs=1)  # back on the floor
    w = np.concatenate([o["wings"] for o in outs if "wings" in o])
    assert w.shape[1] == 6 and np.ptp(w[:, 0]) > 100  # stroke sweeps > 100 deg (flybody wingbeat), both wings alike
    assert np.allclose(w[:, :3], w[:, 3:]) and "wings" not in outs[-1]  # wings folded at the end
    cyc = max((o["wing_cycle"] for o in outs if "wing_cycle" in o), key=lambda c: np.ptp(c[:, 0]))  # one whole wingbeat per chunk, for the viewer's own flap rate
    assert cyc.shape == (16, 6) and np.ptp(cyc[:, 0]) > 100


def test_flight_stays_inside_the_glass_and_legs_tuck():
    b, outs = _fly(bounds=(-8, 8, -8, 8))
    p = np.concatenate([o["thorax"]["pos_um"] for o in outs]) / 1000.0
    assert np.abs(p[:, :2]).max() <= 8.0 + 1e-6 and (p[:, 2] - p[0, 2]).max() <= 45.0
    mid = next(o for o in outs if o["state"] == "airborne" and o["contact"].sum() == 0)
    stand = b.lib.stand.pos
    i = b.lib.bodies.index("lf_tibia")
    c = b.lib.bodies.index("lf_coxa")
    assert np.linalg.norm(mid["frames"][-1][i][:3] - stand[c]) < 0.7 * np.linalg.norm(stand[i] - stand[c])


def test_flight_asset_carries_its_provenance():
    from flybrain.body_kinematic import load_flight
    f = load_flight()
    prov = f.doc["provenance"]
    assert prov["doi"] == "10.25378/janelia.25309105.v4" and prov["license"] == "GPL-3.0-or-later"
    assert {c["kind"] for c in f.clips} == {"straight", "left", "right"} and f.wing.shape == (100, 3)


def _air(asym_l, asym_r, dng02=0.0, land_at_s=None, chunks=200):
    b = KinematicBody()
    run(b, 2, pace=0.0)
    b.takeoff(180.0, "straight")
    outs = []
    for i in range(chunks):
        if b.state in ("airborne", "takeoff"):
            b.set_air(dng02, asym_l, asym_r, land_at_s is not None and i * 0.05 >= land_at_s)
        outs.append(b.advance(50.0, [0.0, 0.0], False, pace=0.0, turn=0.0))
    return b, outs


def test_brain_flight_silent_dns_keep_the_pattern_and_the_dn_asymmetry_steers():
    _, silent = _air(0.0, 0.0, chunks=40)
    w = np.concatenate([o["wings"] for o in silent if "wings" in o])
    assert np.allclose(w[:, :3], w[:, 3:])  # no DN activity: both wings alike
    _, steer = _air(50.0, 0.0, dng02=50.0, chunks=40)
    w2 = np.concatenate([o["wings"] for o in steer if "wings" in o])
    assert np.ptp(w2[:, 0]) > np.ptp(w2[:, 3])  # left DNp03 raises the left stroke amplitude, lowers the right
    yaw = lambda outs: outs[-1]["thorax"]["yaw_deg"][-1]
    assert yaw(steer) < yaw(silent) - 5  # a left DNp03 turns the fly right (Croke's contralateral saccade)


def test_brain_flight_lands_on_the_landing_dns_or_the_max_airtime():
    b, outs = _air(0.0, 0.0, land_at_s=1.0, chunks=100)
    assert b.landing_cause == "landing_dn" and outs[-1]["state"] == "standing"
    b, outs = _air(0.0, 0.0, chunks=260)  # silent landing neurons: the 3-8 s bout ends the flight
    assert b.landing_cause == "bout_end" and outs[-1]["state"] == "standing"
