"""BrainStream.set_rates: per-neuron Poisson rates (what a live eye produces), same machinery as set_stimulus."""

import numpy as np
import pytest

from flybrain.simulation import Simulation, SimulationError

CHAIN = [(62, 61, 1000), (61, 60, 1000), (62, 5, 1000), (60, 0, 1000)]


def chain(n=63):
    pre, post, w = (np.array(c) for c in zip(*CHAIN))
    return Simulation.from_arrays(pre, post, w, n_neurons=n)


def totals(stream, chunks=10):
    out = np.zeros(63, dtype=np.int64)
    for _ in range(chunks):
        c = stream.advance(50.0)
        out[c.active_rows] += c.counts
    return out


def test_a_constant_rate_vector_is_bit_identical_to_set_stimulus():
    a, b = chain().open_stream(seed=2), chain().open_stream(seed=2)
    a.set_stimulus([62, 30], 300.0)
    b.set_rates([62, 30], [300.0, 300.0])
    assert np.array_equal(totals(a), totals(b))


def test_rows_get_their_own_rates_and_zero_rates_drop_out():
    s = chain().open_stream(seed=1)
    s.set_rates([62, 61, 30], [400.0, 0.0, 100.0])
    t = totals(s, 20)
    assert t[62] > t[30] > 0  # the faster one fires more
    assert s.stimulus is not None and list(s.stimulus[0]) == [30, 62]  # zero-rate row 61 is not a Poisson target


def test_rates_can_change_every_chunk_and_none_clears():
    s = chain().open_stream(seed=3)
    s.set_rates([62], [500.0])
    assert 62 in s.advance(100.0).active_rows
    s.set_rates([], [])
    s.advance(100.0)
    assert 62 not in s.advance(100.0).active_rows


def test_validation_leaves_the_stream_as_it_was():
    s = chain().open_stream(seed=0)
    s.set_rates([62], [100.0])
    for rows, rates in (([999], [10.0]), ([62], [-1.0]), ([62], [1e9]), ([62, 61], [1.0]), ([62, 62], [1.0, 2.0]), ([62], [float("nan")])):
        with pytest.raises(SimulationError):
            s.set_rates(rows, rates)
    assert list(s.stimulus[0]) == [62] and s.stimulus[1] == 100.0
