"""LiveSession with vision: the fly's eye feeds the brain each chunk; honest labels in every message."""

import json

import numpy as np
import pytest

from flybrain import fly_eye
from flybrain.fly_eye import ArenaScene, VisionFrontEnd, VisionOut, build_eye_from_columns
from flybrain.live import LiveSession
from flybrain.simulation import ChunkSpikes, SimulationError

GROUPS = {"dng100": np.array([0, 1]), "dna01_left": np.array([2]), "dna01_right": np.array([3]),
          "dna02_left": np.array([4]), "dna02_right": np.array([5]), "dnp01": np.array([6, 7]), "mn9": np.array([8])}


class Brain:
    backend = "cpu"

    def __init__(self):
        self.events = []

    def set_stimulus(self, rows, rate_hz):
        if rows is not None and len(rows) and max(rows) > 100000:
            raise SimulationError("row out of range")
        self.events.append(("stimulus", None if rows is None else list(rows), rate_hz))

    def set_rates(self, rows, rates_hz):
        self.events.append(("rates", [int(r) for r in rows], [float(r) for r in rates_hz]))

    def advance(self, ms):
        self.events.append(("advance", ms))
        return ChunkSpikes(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32), ms)

    def close(self):
        pass


class Body:
    """Walks +x at 10 mm/s; reports its thorax like the real bodies."""

    header = {"format": "fly-walk/1", "bodies": ["a"] * 48}

    def __init__(self, walk=True):
        self.x_um, self.walk = 0.0, walk

    def advance(self, ms, drive, escape):
        n = int(ms / 5)
        xs = self.x_um + (np.arange(1, n + 1) * 50.0 if self.walk else 0.0)  # 50 um per 5 ms frame
        self.x_um = float(xs[-1]) if self.walk else self.x_um
        pos = np.column_stack([xs if self.walk else np.full(n, self.x_um), np.zeros(n), np.full(n, 1022.0)]).astype(int)
        return {"n_frames": n, "frames": np.zeros((n, 48, 7), int), "contact": np.ones((n, 6), int), "escape": np.zeros(n, int),
                "drive": np.zeros((n, 2)), "thorax": {"pos_um": pos, "yaw_deg": np.zeros(n)}}


class FakeVision:
    background_note = None
    background_hz = 0.0

    def __init__(self, rows=(5,), rates=(40.0,)):
        self.calls, self.out = [], (np.asarray(rows, dtype=np.int64), np.asarray(rates, dtype=float))

    def process(self, pos_um, yaw_deg):
        self.calls.append((np.asarray(pos_um).copy(), np.asarray(yaw_deg).copy()))
        return VisionOut(*self.out, {"driven_cells": len(self.out[0]), "max_rate_hz": float(self.out[1].max(initial=0)), "mean_rate_hz": 1.0, "ms": 0.5})


def session(vision=None, brain=None, body=None, **kw):
    brain = brain or Brain()
    return LiveSession(brain, body or Body(), GROUPS, vision=vision, **kw), brain


def test_without_vision_nothing_changes_and_the_message_says_so():
    s, brain = session()
    msg = s.step()
    assert not any(e[0] == "rates" for e in brain.events)
    assert msg["senses"] == {"vision": None, "background_hz": 0.0}
    assert s.start_message()["senses"] == {"vision": None, "background_hz": 0.0} and "vision" not in msg["wall_ms"]


def test_each_chunk_the_eye_sets_the_rates_before_the_brain_advances_and_the_pose_is_the_last_chunk():
    vision = FakeVision(rows=(5, 9), rates=(40.0, 80.0))
    s, brain = session(vision)
    s.step()
    s.step()
    kinds = [e[0] for e in brain.events]
    assert kinds == ["rates", "advance", "rates", "advance"]
    assert list(brain.events[0][1:]) == [[5, 9], [40.0, 80.0]]
    first_pos, second_pos = vision.calls[0][0], vision.calls[1][0]
    assert first_pos.shape == (10, 3) and first_pos[:, 0].max() == 0  # chunk 0 sees the start pose
    assert second_pos[0, 0] == 50 and second_pos[-1, 0] == 500  # chunk 1 sees the frames chunk 0 produced


def test_the_user_stimulus_and_the_eye_add_up_per_neuron():
    s, brain = session(FakeVision(rows=(11, 12), rates=(50.0, 60.0)))
    s.set_stimulus([10, 11], 100.0)
    s.step()
    last = [e for e in brain.events if e[0] == "rates"][-1]
    assert list(last[1:]) == [[10, 11, 12], [100.0, 150.0, 60.0]]


def test_set_stimulus_is_still_validated_with_vision_on():
    s, _ = session(FakeVision())
    with pytest.raises(SimulationError):
        s.set_stimulus([999999], 10.0)


def test_messages_carry_the_honest_senses_block_and_a_caption_about_the_brain():
    vision = FakeVision()
    vision.background_hz, vision.background_note = 2.0, "background noise: 2 Hz on T4/T5"
    s, _ = session(vision)
    started, msg = s.start_message(), s.step()
    assert json.loads(json.dumps(msg)) == msg
    assert msg["senses"]["vision"] == "arena (T4/T5 via HRC; photoreceptor path bypassed)"
    assert msg["senses"]["background_hz"] == 2.0 and "T4/T5" in msg["senses"]["background"]
    assert msg["senses"]["driven_cells"] == 1 and "vision" in msg["wall_ms"]
    assert "whatever the brain does" in started["caption"] and started["senses"]["vision"] == msg["senses"]["vision"]


def test_a_gpu_failure_moves_the_eye_driven_stimulus_to_the_cpu_brain():
    class Gpu(Brain):
        backend = "gpu"

        def advance(self, ms):
            raise RuntimeError("boom")

    cpu = Brain()
    s, _ = session(FakeVision(rows=(5,), rates=(40.0,)), brain=Gpu(), brain_fallback=lambda: cpu)
    s.step()
    assert s.fallback and cpu.events[0][0] == "rates" and list(cpu.events[0][1:]) == [[5], [40.0]]


def test_the_real_eye_sees_the_arena_while_the_fly_walks_and_nothing_while_it_stands():
    pp, qq = (a.ravel() for a in np.meshgrid(np.arange(-10, 11), np.arange(-10, 11)))
    eyes = {"right": build_eye_from_columns("right", pp, qq, first_row=1000), "left": build_eye_from_columns("left", pp, qq, first_row=100000)}
    walking, _ = session(VisionFrontEnd(eyes, ArenaScene()), body=Body(True))
    msgs = [walking.step() for _ in range(6)]
    assert msgs[0]["senses"]["driven_cells"] == 0 and msgs[-1]["senses"]["driven_cells"] > 0
    standing, brain = session(VisionFrontEnd(eyes, ArenaScene()), body=Body(False))
    msgs = [standing.step() for _ in range(6)]
    assert all(m["senses"]["driven_cells"] == 0 for m in msgs)


# -- the background-only front end and the wiring ---------------------------------------------------------------

def tiny_eyes():
    pp, qq = (a.ravel() for a in np.meshgrid(np.arange(-4, 5), np.arange(-4, 5)))
    return {"right": build_eye_from_columns("right", pp, qq, first_row=1000), "left": build_eye_from_columns("left", pp, qq, first_row=100000)}


def test_a_background_only_front_end_sees_nothing_and_says_so():
    v = VisionFrontEnd(tiny_eyes(), ArenaScene(), background_hz=3.0, see=False)
    out = v.process(np.tile([0.0, 0.0, 1022.0], (10, 1)), np.zeros(10))
    assert out.rows.size == v.n_cells and np.allclose(out.rates_hz, 3.0) and out.stats["driven_cells"] == 0
    s, _ = session(v)
    assert s.step()["senses"]["vision"] is None and s.step()["senses"]["background_hz"] == 3.0


class _Sims:
    available = True
    thermal = None
    opened = 0

    def exclusive(self):
        import contextlib
        return contextlib.nullcontext()

    def open_stream(self, seed=0):
        type(self).opened += 1
        return Brain()

    def simulation(self):
        return self


def _build(monkeypatch, eyes, **kw):
    from flybrain import live

    monkeypatch.setattr(live, "readout_groups", lambda atlas, row_index: GROUPS)
    monkeypatch.setattr(fly_eye, "load_eyes", lambda atlas, path=None: eyes)
    _Sims.opened = 0
    return live.build_live_session(object(), None, _Sims(), None, **kw)


def test_build_live_session_wires_the_arena_eye_and_defaults_to_no_vision(monkeypatch):
    assert _build(monkeypatch, tiny_eyes()).vision is None
    s = _build(monkeypatch, tiny_eyes(), vision="arena", background_hz=2.0)
    assert isinstance(s.vision, VisionFrontEnd) and s.vision.background_hz == 2.0 and s.start_message()["senses"]["vision"]
    only_bg = _build(monkeypatch, tiny_eyes(), background_hz=2.0)
    assert only_bg.vision is not None and only_bg.vision.sees is False


def test_a_missing_column_file_fails_before_anything_heavy_is_built(monkeypatch):
    with pytest.raises(fly_eye.VisionUnavailable, match="column"):
        _build(monkeypatch, None, vision="arena")
    assert _Sims.opened == 0
    with pytest.raises(ValueError, match="vision"):
        _build(monkeypatch, tiny_eyes(), vision="retina")
    with pytest.raises(ValueError, match="background_hz"):
        _build(monkeypatch, tiny_eyes(), background_hz=-1.0)
