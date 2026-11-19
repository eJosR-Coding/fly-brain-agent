"""LiveSession state machine with a fake brain and a fake body: message schema, drive mapping, speed, stimulus, thermal."""

import json

import numpy as np
import pytest

from flybrain.gpu_simulation import GPUCapacityError, ThermalGuard
from flybrain import live
from flybrain.live import LiveSession, frames_from_bytes, frames_to_bytes
from flybrain.simulation import ChunkSpikes, SimulationError

GROUPS = {"dng100": np.array([0, 1]), "dna01_left": np.array([2]), "dna01_right": np.array([3]),
          "dna02_left": np.array([4]), "dna02_right": np.array([5]), "dnp01": np.array([6, 7]), "mn9": np.array([8])}


class Clock:
    """perf_counter stand-in: each call returns the current time; fakes move it to simulate compute."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeBrain:
    def __init__(self, clock, backend="cpu", cost_s=0.01, script=None, fail=None):
        self.clock, self.backend, self.cost_s = clock, backend, cost_s
        self.script, self.fail = script or {}, fail
        self.stimuli, self.advances, self.closed, self.i = [], [], False, 0

    def set_stimulus(self, rows, rate_hz):
        if rows is not None and len(rows) and max(rows) > 100:
            raise SimulationError("row out of range")
        self.stimuli.append((None if rows is None else list(rows), rate_hz))

    def advance(self, ms):
        if self.fail:
            raise self.fail
        self.clock.t += self.cost_s
        self.advances.append(ms)
        spikes = self.script.get(self.i, {})
        self.i += 1
        rows = np.array(sorted(spikes), dtype=np.int64)
        return ChunkSpikes(rows, np.array([spikes[r] for r in rows], dtype=np.int32), ms)

    def close(self):
        self.closed = True


class FakeBody:
    def __init__(self, clock, cost_s=0.2):
        self.clock, self.cost_s, self.calls = clock, cost_s, []
        self.header = {"format": "fly-walk/1", "bodies": ["a"] * 48}

    def advance(self, ms, drive, escape):
        self.clock.t += self.cost_s
        self.calls.append((ms, list(drive), escape))
        n = int(ms / 5)
        return {"n_frames": n, "frames": np.arange(n * 48 * 7).reshape(n, 48, 7) % 9000 - 4500,
                "contact": np.ones((n, 6), int), "escape": np.full(n, int(escape)),
                "drive": np.tile(np.array(drive, float), (n, 1)) * (0 if escape else 1),
                "thorax": {"pos_um": np.zeros((n, 3), int), "yaw_deg": np.zeros(n)}}


def make(clock=None, **kw):
    clock = clock or Clock()
    brain = kw.pop("brain", None) or FakeBrain(clock)
    body = kw.pop("body", None) or FakeBody(clock)
    return LiveSession(brain, body, GROUPS, clock=clock, **kw), brain, body, clock


def walking_script(n=20):
    # 40 Hz on both DNg100 neurons: 2 spikes each per 50 ms chunk (below the saturation after smoothing)
    return {i: {0: 2, 1: 2} for i in range(n)}


def test_chunk_message_schema_is_json_serialisable():
    s, *_ = make()
    msg = s.step()
    assert json.loads(json.dumps(msg)) == msg
    assert msg["type"] == "chunk" and msg["seq"] == 0 and msg["t_ms"] == 50.0 and msg["chunk_ms"] == 50.0
    assert set(msg["dn"]) == set(GROUPS) and isinstance(msg["escape"], bool) and len(msg["drive"]) == 2
    assert msg["backend"] == "cpu" and msg["fallback"] is None and msg["active_count"] == 0
    assert len(msg["frames"]) == 10 and len(msg["frames"][0]) == 48 and len(msg["frames"][0][0]) == 7
    assert set(msg["thorax"]) == {"pos_um", "yaw_deg"} and len(msg["thorax"]["pos_um"]) == 10
    assert set(msg["wall_ms"]) == {"brain", "body", "total"} and "speed_factor" in msg and "speed_label" in msg
    assert msg["stimulus"] is None


def test_seq_and_simulated_time_accumulate_and_chunk_ms_reaches_both_halves():
    s, brain, body, _ = make()
    msgs = [s.step() for _ in range(3)]
    assert [m["seq"] for m in msgs] == [0, 1, 2] and [m["t_ms"] for m in msgs] == [50.0, 100.0, 150.0]
    assert brain.advances == [50.0] * 3 and [c[0] for c in body.calls] == [50.0] * 3


def test_dn_rates_are_group_means_in_hz_and_active_count_counts_active_neurons():
    clock = Clock()
    brain = FakeBrain(clock, script={0: {0: 5, 1: 5, 2: 2, 8: 1, 20: 3}})
    s, *_ = make(clock, brain=brain)
    msg = s.step()
    assert msg["dn"]["dng100"] == pytest.approx(100.0) and msg["dn"]["dna01_left"] == pytest.approx(40.0)
    assert msg["dn"]["mn9"] == pytest.approx(20.0) and msg["dn"]["dnp01"] == 0.0
    assert msg["active_count"] == 5


def test_brain_rates_drive_the_body_with_smoothing_carried_between_chunks():
    clock = Clock()
    s, brain, body, _ = make(clock, brain=FakeBrain(clock, script=walking_script()))
    first, second = s.step(), s.step()
    assert body.calls[0][1] == pytest.approx(first["drive"], abs=1e-4) and first["drive"][0] > 0
    assert first["forward"] < second["forward"] < 1.0  # the low-pass is still charging
    assert not first["escape"] and body.calls[0][2] is False


def test_escape_neuron_stops_the_drive_and_raises_the_flag():
    clock = Clock()
    brain = FakeBrain(clock, script={i: {0: 2, 1: 2, 6: 20, 7: 20} for i in range(5)})
    s, _, body, _ = make(clock, brain=brain)
    msgs = [s.step() for _ in range(4)]
    assert msgs[-1]["escape"] is True and msgs[-1]["drive"] == [0.0, 0.0] and body.calls[-1][2] is True


def test_speed_is_measured_never_pretended():
    clock = Clock()
    s, *_ = make(clock, brain=FakeBrain(clock, cost_s=0.01), body=FakeBody(clock, cost_s=0.2))
    msg = s.step()
    assert msg["wall_ms"]["brain"] == pytest.approx(10.0) and msg["wall_ms"]["body"] == pytest.approx(200.0)
    assert msg["speed_factor"] == pytest.approx(50.0 / 210.0, abs=1e-3)
    assert msg["speed_label"] == "x0.24 real time"


def test_speed_label_for_faster_than_real_time():
    clock = Clock()
    s, *_ = make(clock, brain=FakeBrain(clock, cost_s=0.005), body=FakeBody(clock, cost_s=0.01))
    assert s.step()["speed_label"] == "x3.33 real time"


def test_stimulus_is_forwarded_summarised_and_validated():
    s, brain, *_ = make()
    s.set_stimulus([10, 11, 12], 100.0)
    assert brain.stimuli[-1] == ([10, 11, 12], 100.0)
    assert s.step()["stimulus"] == {"rows": 3, "rate_hz": 100.0}
    s.set_stimulus(None, 0.0)
    assert s.step()["stimulus"] is None
    with pytest.raises(SimulationError):
        s.set_stimulus([999], 10.0)
    assert s.step()["stimulus"] is None  # the failed change left the session as it was


def test_hot_gpu_moves_the_session_to_the_cpu_and_says_so():
    clock = Clock()
    gpu = FakeBrain(clock, backend="gpu", script=walking_script())
    cpu = FakeBrain(clock, backend="cpu", script=walking_script())
    temps = iter([70.0, 85.0, 60.0])
    guard = ThermalGuard(read=lambda: {"temperature_c": next(temps)}, interval_s=0.0, clock=clock)
    s, _, _, _ = make(clock, brain=gpu, thermal=guard, brain_fallback=lambda: cpu)
    s.set_stimulus([10], 120.0)
    assert s.step()["backend"] == "gpu"
    msg = s.step()  # 85 C: refuse new GPU work
    assert msg["backend"] == "cpu" and "hot" in msg["fallback"].lower() and "85" in msg["fallback"]
    assert gpu.closed and cpu.stimuli[-1] == ([10], 120.0)  # the stimulus follows the session
    later = s.step()  # cooled to 60 C, but the session stays on the CPU
    assert later["backend"] == "cpu" and later["fallback"] == msg["fallback"] and len(cpu.advances) == 2


def test_gpu_failure_falls_back_to_the_cpu_but_a_cpu_failure_is_raised():
    clock = Clock()
    cpu = FakeBrain(clock)
    s, *_ = make(clock, brain=FakeBrain(clock, backend="gpu", fail=GPUCapacityError("no room")),
                 brain_fallback=lambda: cpu)
    msg = s.step()
    assert msg["backend"] == "cpu" and "no room" in msg["fallback"]
    broken, *_ = make(brain=FakeBrain(Clock(), fail=RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        broken.step()


def test_binary_frames_roundtrip_and_leave_the_json_light():
    s, *_ = make(binary_frames=True)
    msg, blob = s.step_binary()
    assert msg["frames"] is None and msg["frames_binary"] == {"dtype": "<i2", "shape": [10, 48, 7]}
    assert len(blob) == 10 * 48 * 7 * 2
    plain = make()[0].step()
    assert np.array_equal(frames_from_bytes(blob, (10, 48, 7)), np.array(plain["frames"]))
    assert len(json.dumps(msg)) < len(json.dumps(plain)) / 5


def test_frames_to_bytes_clips_to_int16():
    out = frames_from_bytes(frames_to_bytes(np.array([[[40000, -40000]]])), (1, 1, 2))
    assert out.tolist() == [[[32767, -32768]]]


def test_active_rows_are_opt_in_and_capped_by_count():
    clock = Clock()
    brain = FakeBrain(clock, script={0: {r: (r % 7) + 1 for r in range(30)}})
    off, *_ = make(clock, brain=brain)
    assert "active" not in off.step()
    clock2 = Clock()
    on, *_ = make(clock2, brain=FakeBrain(clock2, script={0: {r: (r % 7) + 1 for r in range(30)}}),
                  include_active=True, max_active=5)
    a = on.step()["active"]
    assert len(a["rows"]) == 5 == len(a["counts"]) and min(a["counts"]) == 6 and a["total"] == 30


def test_close_closes_the_brain_once_and_step_after_close_fails():
    s, brain, *_ = make()
    s.close()
    s.close()
    assert brain.closed
    with pytest.raises(RuntimeError):
        s.step()


def test_header_message_describes_the_clip_format_and_limits():
    s, brain, body, _ = make()
    h = s.start_message()
    assert h["type"] == "started" and h["chunk_ms"] == 50.0 and h["backend"] == "cpu"
    assert h["header"] == body.header and set(h["dn_names"]) == set(GROUPS)
    assert "our" in h["caption"].lower()


def test_body_mode_and_caption_are_reported_in_started_and_every_chunk():
    s, *_ = make()  # the fake body has no mode: reported as the physics default
    started = s.start_message()
    assert started["body_mode"] == "physics" and "body_caption" in started
    msg = s.step()
    assert msg["body_mode"] == "physics" and "body_caption" in msg and "world" not in msg


def test_a_body_that_takes_the_command_gets_the_exact_pace_and_turn():
    class CommandBody(FakeBody):
        mode, takes_command, caption = "kinematic", True, "recorded physics gait, blended live"
        commands: list = []

        def advance(self, ms, drive, escape, *, pace=None, turn=None):
            self.commands.append((pace, turn))
            return super().advance(ms, drive, escape)

    clock = Clock()
    s, brain, body, _ = make(clock, brain=FakeBrain(clock, script=walking_script()), body=CommandBody(clock))
    body.commands = []
    for _ in range(3):
        msg = s.step()
    assert all(p is not None and t == 0.0 for p, t in body.commands) and body.commands[-1][0] > 0
    assert msg["body_mode"] == "kinematic" and msg["body_caption"].startswith("recorded physics gait")


# -- arena: the server owns the glass cube ---------------------------------------------------

def test_arena_spec_is_centred_on_the_start_and_documents_the_world_rule():
    from flybrain.live import ARENA_INTERIOR_MM, arena_spec

    a = arena_spec()
    h = ARENA_INTERIOR_MM / 2
    assert a["kind"] == "glass_cube" and a["interior_mm"] == ARENA_INTERIOR_MM == 50.0
    assert a["bounds_mm"] == [-h, h, -h, h]  # xmin, xmax, ymin, ymax; x forward, y left (fly frame at the start)
    assert "world rule" in a["note"].lower() and "not a brain decision" in a["note"].lower()
    assert arena_spec(20.0)["bounds_mm"] == [-10.0, 10.0, -10.0, 10.0]


def test_started_carries_the_arena_and_is_json_serialisable():
    from flybrain.live import arena_spec

    s, *_ = make(arena=arena_spec())
    started = s.start_message()
    json.dumps(started)
    assert started["arena"]["kind"] == "glass_cube" and started["arena"]["bounds_mm"] == [-25.0, 25.0, -25.0, 25.0]
    assert make()[0].start_message()["arena"] is None


def test_build_live_session_passes_the_arena_bounds_to_the_kinematic_body(monkeypatch):
    from flybrain import live
    from flybrain.body_kinematic import KinematicBody

    class Sims:
        available = True
        thermal = None

        def exclusive(self):
            import contextlib
            return contextlib.nullcontext()

        def open_stream(self, seed=0):
            return FakeBrain(Clock())

        def simulation(self):
            return self

    class Atlas:
        pass

    monkeypatch.setattr(live, "readout_groups", lambda atlas, row_index: GROUPS)
    s = live.build_live_session(Atlas(), None, Sims(), None)
    assert isinstance(s.body, KinematicBody) and s.body.bounds == (-25.0, 25.0, -25.0, 25.0)
    assert s.start_message()["arena"]["bounds_mm"] == [-25.0, 25.0, -25.0, 25.0]


def test_a_fake_brain_driving_forward_walks_onto_the_wall_and_the_chunk_says_which_surface():
    from flybrain.body_kinematic import KinematicBody, load_library
    from flybrain.live import arena_spec

    clock = Clock()
    a = arena_spec(10.0)  # small cube: the wall is reached within a few seconds
    body = KinematicBody(load_library(), bounds=tuple(a["bounds_mm"]), height_mm=a["interior_mm"])
    s, *_ = make(clock, brain=FakeBrain(clock, script=walking_script(200)), body=body, arena=a)
    first, last, xs = None, None, []
    for _ in range(120):
        m = s.step()
        first = first or m
        last = m
        xs.extend(p[0] for p in m["thorax"]["pos_um"])
    assert first["world"]["surface"] == "floor" and last["world"]["surface"] != "floor"
    assert max(xs) <= 5000 and last["world"]["bounds_mm"] == [-5.0, 5.0, -5.0, 5.0]
    assert "world rule" in last["world"]["note"].lower()


def test_physics_mode_clamps_the_reported_root_and_reports_the_wall():
    from flybrain.live import arena_spec

    class Walker(FakeBody):  # a body that does not enforce walls (like the MuJoCo one): walks +x 2 mm per chunk
        def __init__(self, clock):
            super().__init__(clock)
            self.x = 0

        def advance(self, ms, drive, escape):
            out = super().advance(ms, drive, escape)
            n = out["n_frames"]
            xs = self.x + np.arange(1, n + 1) * 200
            self.x = int(xs[-1])
            out["thorax"] = {"pos_um": np.stack([xs, np.zeros(n, int), np.zeros(n, int)], 1), "yaw_deg": np.zeros(n)}
            return out

    clock = Clock()
    s, *_ = make(clock, body=Walker(clock), arena=arena_spec(10.0))
    msgs = [s.step() for _ in range(4)]
    xs = [p[0] for m in msgs for p in m["thorax"]["pos_um"]]
    assert max(xs) == 5000 and msgs[-1]["world"]["clamped_walls"] == ["x_max"] and msgs[-1]["world"]["surface"] == "floor"
    assert msgs[0]["world"]["clamped_walls"] == []
    assert "display" in msgs[-1]["world"]["note"].lower()  # honest: the clamp is of the reported root, not physics


def test_a_touched_cube_drives_its_taste_neurons_and_is_reported_in_the_chunk():
    from flybrain import live_world

    s, *_ = make()
    s.brain.set_rates = lambda rows, rates: s.brain.stimuli.append((list(rows), list(rates)))
    s.taste_rows = {"sugar": np.array([1, 2]), "bitter": np.array([3])}
    s.world_add("sugar", ahead_mm=2.0)
    msg = s.step()
    assert msg["world"]["objects"][0]["kind"] == "sugar"
    assert msg["world"]["touching"] == ["sugar1"] and msg["world"]["taste_hz"] == {"sugar": live_world.TASTE_HZ}
    assert s.world_clear() == 1
    assert s.step()["world"]["touching"] == []


def test_touch_sides_follow_the_wall_relative_to_the_heading():
    assert live.touch_sides(["x_max"], 0.0) == ["left", "right"]  # head-on
    assert live.touch_sides(["y_max"], 0.0) == ["left"]  # wall on the fly's left
    assert live.touch_sides(["y_min"], 0.0) == ["right"]
    assert live.touch_sides(["x_min"], 0.0) == []  # wall behind: no antenna
    assert live.touch_sides(["x_max"], 90.0) == ["right"]  # heading +y, wall at +x is on its right
    assert live.touch_sides([], 0.0) == []


def test_world_threat_is_a_moving_object_and_leaves_after_its_life():
    from flybrain import live_world

    s, *_ = make()
    o = s.world_threat("right")
    assert o["kind"] == "threat" and o["size_mm"] == 2 * live_world.THREAT_R_MM
    first = s.step()["world"]["objects"][0]["pos_mm"]
    assert s.step()["world"]["objects"][0]["pos_mm"] != first  # it moves
    with pytest.raises(ValueError):
        s.world_threat("up")
    for _ in range(40):
        msg = s.step()
    assert msg["world"]["objects"] == []  # life over (~1.1 s)
    assert "state" not in msg  # the fake body has no takeoff


def test_moving_sphere_darkens_the_rays_it_covers_and_grows():
    from flybrain import fly_eye

    scene = fly_eye.ArenaScene()
    dirs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    base = scene.luminance(np.array([[0.0, 0.0, 1.3]]), np.zeros(1), dirs)
    scene.moving = [fly_eye.MovingSphere((10.0, 0.0, 1.3), (-25.0, 0.0, 0.0), 1.0, 0.0)]
    lum = scene.luminance(np.array([[0.0, 0.0, 1.3]]), np.zeros(1), dirs)
    assert lum[0, 0] < 0.2 * base[0, 0] and abs(lum[0, 1] - base[0, 1]) < 1e-6
