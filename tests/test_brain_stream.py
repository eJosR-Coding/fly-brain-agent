"""Streaming brain (CPU): the LIF state lives between advance() calls, so a run is a sequence of chunks."""

import numpy as np
import pytest

from flybrain.simulation import ChunkSpikes, Simulation, SimulationError

CHAIN = [(62, 61, 1000), (61, 60, 1000), (62, 5, 1000), (60, 0, 1000)]


def chain(n=63):
    pre, post, w = (np.array(c) for c in zip(*CHAIN))
    return Simulation.from_arrays(pre, post, w, n_neurons=n)


def dense(chunk: ChunkSpikes, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.int64)
    out[chunk.active_rows] = chunk.counts
    return out


def test_chunked_advance_is_bit_identical_to_one_run_on_the_cpu():
    sim = chain()
    ref = sim.run([62], 300.0, duration_ms=1000, trials=1, seed=3)
    stream = sim.open_stream(seed=3)
    stream.set_stimulus([62], 300.0)
    total = np.zeros(sim.n, dtype=np.int64)
    for _ in range(20):
        total += dense(stream.advance(50.0), sim.n)
    assert np.array_equal(total, ref.counts) and ref.counts.sum() > 0
    assert stream.t_ms == pytest.approx(1000.0)


def test_chunk_shape_sorted_active_rows_positive_counts_and_duration():
    sim = chain()
    s = sim.open_stream(seed=0)
    s.set_stimulus([62], 300.0)
    chunk = s.advance(100.0)
    assert chunk.duration_ms == 100.0
    assert np.all(np.diff(chunk.active_rows) > 0) and np.all(chunk.counts > 0)
    assert chunk.active_rows.dtype == np.int64 and len(chunk.active_rows) == len(chunk.counts)


def test_group_hz_is_mean_rate_of_the_group_in_the_chunk():
    chunk = ChunkSpikes(np.array([2, 5], dtype=np.int64), np.array([4, 1], dtype=np.int32), 50.0)
    # rows 2 and 3: 4 spikes over 2 neurons in 50 ms = 40 Hz each; a missing row counts as silent
    assert chunk.group_hz([2, 3]) == pytest.approx(40.0)
    assert chunk.group_hz([9]) == 0.0 and chunk.group_hz([]) == 0.0


def test_no_stimulus_means_silence_and_stimulus_can_change_mid_session():
    sim = chain()
    s = sim.open_stream(seed=1)
    assert len(s.advance(50.0).active_rows) == 0
    s.set_stimulus([62], 400.0)
    assert 62 in s.advance(100.0).active_rows
    s.set_stimulus(None, 0.0)
    s.advance(100.0)  # let in-flight synaptic input die out
    quiet = s.advance(100.0)
    assert 62 not in quiet.active_rows


def test_stimulus_change_is_validated_and_leaves_the_session_usable():
    s = chain().open_stream()
    with pytest.raises(SimulationError):
        s.set_stimulus([999], 100.0)
    with pytest.raises(SimulationError):
        s.set_stimulus([62], -1.0)
    with pytest.raises(SimulationError):
        s.advance(0.05)  # not a multiple of dt
    s.set_stimulus([62], 100.0)
    s.advance(10.0)


def test_streams_are_independent_and_seeded():
    sim = chain()
    a, b, c = sim.open_stream(seed=1), sim.open_stream(seed=1), sim.open_stream(seed=2)
    for st in (a, b, c):
        st.set_stimulus([62], 200.0)
    ra, rb, rc = (dense(st.advance(200.0), sim.n) for st in (a, b, c))
    assert np.array_equal(ra, rb) and not np.array_equal(ra, rc)


def test_silenced_rows_do_not_propagate():
    sim = chain()
    s = sim.open_stream(seed=0, silenced=[62])
    s.set_stimulus([62], 400.0)
    out = dense(s.advance(300.0), sim.n)
    assert out[62] > 0 and out[61] == 0 and out[5] == 0
