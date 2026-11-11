"""Time-varying stimulus: per-neuron rate schedule (piecewise-constant windows) on tiny networks."""

import numpy as np
import pytest

from flybrain.sim_service import SimulationService
from flybrain.simulation import SimulationError, Simulation, StimulusSchedule


def tiny(n=6):
    # 0 -> 1 -> 2 chain, 3/4/5 isolated sensory-ish neurons
    return Simulation.from_arrays([0, 1], [1, 2], [300, 300], n_neurons=n)


def test_constant_run_is_pinned():
    """Regression: constant stimulation must keep producing exactly these numbers (captured before the schedule work)."""
    sim = Simulation.from_arrays([0, 0, 1], [1, 2, 2], [300, 200, 200], n_neurons=4)
    r = sim.run([0, 3], 300.0, duration_ms=100, trials=2, seed=7, bin_ms=10.0)
    assert r.counts.tolist() == [69, 30, 30, 56]
    assert r.trial_counts.tolist() == [[36, 15, 15, 32], [33, 15, 15, 24]]
    assert r.frames.tolist() == [
        [6, 8, 10, 5, 6, 4, 8, 4, 11, 7], [2, 3, 3, 4, 3, 3, 3, 3, 3, 3],
        [1, 4, 3, 3, 3, 2, 4, 3, 3, 4], [3, 8, 6, 8, 6, 6, 2, 7, 6, 4],
    ]


def test_full_length_segment_equals_constant_run_bit_for_bit():
    sim = Simulation.from_arrays([0, 0, 1], [1, 2, 2], [300, 200, 200], n_neurons=4)
    const = sim.run([0, 3], 300.0, duration_ms=100, trials=2, seed=7, bin_ms=10.0)
    sched = StimulusSchedule([(0, 100, [3, 0], 300.0)])
    res = sim.run_schedule(sched, duration_ms=100, trials=2, seed=7, bin_ms=10.0)
    assert np.array_equal(res.trial_counts, const.trial_counts)
    assert np.array_equal(res.frames, const.frames) and np.array_equal(res.active_rows, const.active_rows)


def test_input_only_inside_the_window():
    sim = tiny()
    sched = StimulusSchedule([(40, 80, [3], 400.0)])
    res = sim.run_schedule(sched, duration_ms=120, seed=1, bin_ms=10.0)
    per_bin = res.binned[:, 3]
    assert per_bin[:4].sum() == 0 and per_bin[8:].sum() == 0  # silent before 40 ms and after 80 ms (no leftover kicks)
    assert per_bin[4:8].sum() > 0


def test_neurons_in_different_windows_do_not_leak():
    sim = tiny()
    sched = StimulusSchedule([(0, 50, [3], 400.0), (50, 100, [4], 400.0)])
    b = sim.run_schedule(sched, duration_ms=100, seed=2, bin_ms=10.0).binned
    assert b[:5, 3].sum() > 0 and b[5:, 3].sum() == 0
    assert b[:5, 4].sum() == 0 and b[5:, 4].sum() > 0


def test_growing_schedule_grows_activity():
    sim = tiny(30)
    rows = list(range(3, 30))
    sched = StimulusSchedule([(0, 100, rows[:3], 200.0), (100, 200, rows[:12], 200.0), (200, 300, rows, 200.0)])
    pop = sim.run_schedule(sched, duration_ms=300, trials=3, seed=3, bin_ms=100.0).population
    assert pop[0] < pop[1] < pop[2]


def test_overlapping_segments_add_up():
    sim = tiny()
    two = StimulusSchedule([(0, 100, [3], 150.0), (0, 100, [3], 150.0)])
    one = StimulusSchedule([(0, 100, [3], 300.0)])
    a = sim.run_schedule(two, duration_ms=100, seed=4)
    b = sim.run_schedule(one, duration_ms=100, seed=4)
    assert np.array_equal(a.counts, b.counts)


@pytest.mark.parametrize("segments, match", [
    ([], "at least one segment"),
    ([(50, 50, [3], 100.0)], "start_ms < end_ms"),
    ([(-10, 50, [3], 100.0)], "start_ms"),
    ([(0, 50, [], 100.0)], "rows"),
    ([(0, 50, [3], -1.0)], "rate_hz"),
    ([(0, 50, [3], float("nan"))], "rate_hz"),
    ([(0, 50, [3], 1e9)], "rate_hz"),
])
def test_schedule_validation_is_clear(segments, match):
    with pytest.raises(SimulationError, match=match):
        StimulusSchedule(segments)


def test_run_schedule_rejects_bad_context():
    sim = tiny()
    with pytest.raises(SimulationError, match="duration"):
        sim.run_schedule(StimulusSchedule([(0, 200, [3], 100.0)]), duration_ms=100)
    with pytest.raises(SimulationError, match="row"):
        sim.run_schedule(StimulusSchedule([(0, 50, [99], 100.0)]), duration_ms=100)
    with pytest.raises(SimulationError, match="multiple"):
        sim.run_schedule(StimulusSchedule([(0, 50.03, [3], 100.0)]), duration_ms=100)


def test_canonical_form_ignores_order_and_duplicates():
    a = StimulusSchedule([(0, 50, [3, 4], 100.0), (50, 100, [5], 80.0)])
    b = StimulusSchedule([(50, 100, [5, 5], 80.0), (0, 50, [4, 3, 3], 100.0)])
    assert a == b and hash(a) == hash(b) and a.key == b.key


# -- service cache ----------------------------------------------------------------------------

@pytest.fixture
def svc():
    return SimulationService.from_simulation(tiny())


def sched_run(svc, segments, **kw):
    kw.setdefault("duration_ms", 100)
    kw.setdefault("seed", 0)
    with svc.exclusive():
        return svc.run_schedule(StimulusSchedule(segments), **kw)


def test_same_schedule_hits_and_is_identical(svc):
    seg = [(0, 50, [3], 200.0), (50, 100, [3, 4], 200.0)]
    a, hit1 = sched_run(svc, seg, bin_ms=10.0)
    b, hit2 = sched_run(svc, list(reversed(seg)), bin_ms=10.0)
    assert (hit1, hit2) == (False, True) and np.array_equal(a.frames, b.frames)


@pytest.mark.parametrize("other", [
    [(0, 50, [3], 200.0), (50, 100, [3, 5], 200.0)],  # different rows
    [(0, 50, [3], 200.0), (50, 100, [3, 4], 210.0)],  # different rate
    [(0, 40, [3], 200.0), (40, 100, [3, 4], 200.0)],  # different window
])
def test_different_segment_misses(svc, other):
    sched_run(svc, [(0, 50, [3], 200.0), (50, 100, [3, 4], 200.0)])
    assert sched_run(svc, other)[1] is False


def test_schedule_and_constant_runs_do_not_share_entries(svc):
    """Documented choice: a constant run and the equivalent one-segment schedule are separate cache entries."""
    with svc.exclusive():
        svc.run([3], 200.0, duration_ms=100, seed=0)
    assert sched_run(svc, [(0, 100, [3], 200.0)])[1] is False
