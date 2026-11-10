"""Tiny-network tests for the LIF simulator (no real connectome, no Brian2)."""

import numpy as np
import pytest

from flybrain.simulation import LIFParams, Simulation

DT = 0.1  # ms, the reference Brian2 default clock
FAST = 10_000.0  # Hz: rate * dt == 1, so the Poisson input fires on EVERY step (deterministic)


def net(edges, n):
    """Simulation from (pre, post, signed synapse count) triples."""
    pre, post, w = (np.array(c) for c in zip(*edges)) if edges else ([], [], [])
    return Simulation.from_arrays(pre, post, w, n_neurons=n)


def spike_steps(res, row):
    """Step indices (0.1 ms each) at which `row` spiked; needs bin_ms == dt."""
    return np.flatnonzero(res.binned[:, row]).tolist()


def test_no_input_no_spikes():
    sim = net([(0, 1, 50), (1, 2, 50)], 3)
    res = sim.run([], 100.0, duration_ms=200, trials=2, seed=1)
    assert res.counts.sum() == 0
    assert res.rate_hz.shape == (3,)


def test_matches_reference_spike_trains():
    """Spike trains recorded from the reference Brian2 model (v_0 = -52 mV, dt = 0.1 ms) on the
    same 4-neuron net: A (Poisson-driven every step) -> B (1000), C (10), D (-10)."""
    sim = net([(0, 1, 1000), (0, 2, 10), (0, 3, -10)], 4)
    res = sim.run([0], FAST, duration_ms=30, trials=1, seed=0, bin_ms=DT)
    assert spike_steps(res, 0) == list(range(1, 300, 2))
    assert spike_steps(res, 1) == [23, 49, 75, 101, 127, 153, 179, 205, 231, 257, 283]
    assert spike_steps(res, 2) == [75, 153, 231]
    assert spike_steps(res, 3) == []


def test_constant_drive_spike_timing_is_analytic():
    """Isolated neuron, sub-threshold kicks every step: first spike where the exact discrete
    recurrence (decay, then kick) first crosses threshold."""
    p = LIFParams(w_syn=0.01)  # kick = 2.5 mV per step
    sim = Simulation.from_arrays([], [], [], n_neurons=1, params=p)
    res = sim.run([0], FAST, duration_ms=10, trials=1, seed=0, bin_ms=DT)
    decay = np.exp(-DT / p.t_mbr)
    v, expected = p.v0, None
    for k in range(100):
        v = p.v0 + (v - p.v0) * decay  # state update at tick k (g stays 0)
        if v > p.v_th:
            expected = k
            break
        v += 2.5  # Poisson kick at tick k
    assert expected is not None
    assert spike_steps(res, 0)[0] == expected


def test_refractory_period_blocks_respiking():
    sim = net([(0, 1, 1000)], 2)
    res = sim.run([0], FAST, duration_ms=100, trials=1, seed=0, bin_ms=DT)
    gaps = np.diff(spike_steps(res, 1))
    assert len(gaps) > 3
    assert gaps.min() >= 22  # 2.2 ms
    assert np.diff(spike_steps(res, 0)).min() == 2  # Poisson-driven neurons have no refractory


def test_synaptic_delay_is_1_8_ms():
    sim = net([(0, 1, 1000)], 2)
    res = sim.run([0], FAST, duration_ms=10, trials=1, seed=0, bin_ms=DT)
    first_a, first_b = spike_steps(res, 0)[0], spike_steps(res, 1)[0]
    assert first_b > first_a + 18  # nothing happens before the 18-step delay has passed
    assert first_b == first_a + 22  # and the response is reproducible (reference value)


def test_inhibition_lowers_drive():
    exc = net([(0, 2, 10)], 3)
    both = net([(0, 2, 10), (1, 2, -10)], 3)
    base = exc.run([0], 300.0, duration_ms=1000, trials=3, seed=3).counts[2]
    inhibited = both.run([0, 1], 300.0, duration_ms=1000, trials=3, seed=3).counts[2]
    assert base > 0
    assert inhibited < base


def test_poisson_rate_statistics():
    sim = net([], 1)
    res = sim.run([0], 100.0, duration_ms=1000, trials=20, seed=5)
    assert res.rate_hz[0] == pytest.approx(100.0, rel=0.1)  # each event makes one spike


def test_silencing_removes_downstream_effect():
    sim = net([(0, 1, 1000), (1, 2, 1000)], 3)
    free = sim.run([0], 100.0, duration_ms=500, trials=2, seed=2)
    cut = sim.run([0], 100.0, duration_ms=500, trials=2, seed=2, silence_rows=[1])
    assert free.counts[2] > 0
    assert cut.counts[1] == free.counts[1] > 0  # the silenced neuron itself still spikes
    assert cut.counts[2] == 0


def test_seed_determinism():
    sim = net([(0, 1, 40)], 2)
    a = sim.run([0], 100.0, duration_ms=300, trials=2, seed=7)
    b = sim.run([0], 100.0, duration_ms=300, trials=2, seed=7)
    c = sim.run([0], 100.0, duration_ms=300, trials=2, seed=8)
    assert (a.counts == b.counts).all()
    assert not (a.counts == c.counts).all()


def test_binned_counts_sum_to_totals():
    sim = net([(0, 1, 1000)], 2)
    res = sim.run([0], 200.0, duration_ms=100, trials=2, seed=4, bin_ms=10)
    assert res.binned.shape == (10, 2)
    assert (res.binned.sum(axis=0) == res.counts).all()
    assert res.population.tolist() == res.binned.sum(axis=1).tolist()


@pytest.mark.parametrize(
    "kwargs",
    [dict(stim_rows=[5], rate_hz=10), dict(stim_rows=[0], rate_hz=-1), dict(stim_rows=[0], rate_hz=20_000),
     dict(stim_rows=[0], rate_hz=10, silence_rows=[9]), dict(stim_rows=[0], rate_hz=10, trials=0),
     dict(stim_rows=[0], rate_hz=10, bin_ms=0.05)],
)
def test_bad_arguments_raise(kwargs):
    sim = net([(0, 1, 5)], 2)
    with pytest.raises(ValueError):
        sim.run(**kwargs)


def test_from_arrays_rejects_out_of_range_rows():
    with pytest.raises(ValueError):
        Simulation.from_arrays([0], [3], [1], n_neurons=2)
