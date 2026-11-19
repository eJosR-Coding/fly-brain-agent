"""The active-set GPU path (single trial) must match the dense per-tick kernel exactly: skipping a neuron whose
update is a bit-identical fixed point cannot change any spike. Skipped without CuPy and a CUDA device."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy", reason="CuPy (the `gpu` extra) is not installed")

from flybrain import gpu_simulation as gs  # noqa: E402
from flybrain.simulation import Simulation  # noqa: E402

try:
    HAVE_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # noqa: BLE001
    HAVE_GPU = False
pytestmark = pytest.mark.skipif(not HAVE_GPU, reason="no CUDA device")


@pytest.fixture(autouse=True)
def _free_pool():
    yield
    cp.get_default_memory_pool().free_all_blocks()


def ring_net(n=400, fan=6, seed=3):
    """Every neuron has `fan` outgoing edges to distinct targets with equal weights: float sums are order-free enough
    that two runs on the same stream must agree exactly."""
    rng = np.random.default_rng(seed)
    pre = np.repeat(np.arange(n), fan)
    post = np.concatenate([rng.choice(n, fan, replace=False) for _ in range(n)])
    w = np.full(pre.size, 500)
    return Simulation.from_arrays(pre, post, w, n_neurons=n)


def stream_counts(cpu, active: bool, schedule, seed=11):
    gpu = gs.GPUSimulation.from_simulation(cpu)
    gpu.active_set = active
    s = gpu.open_stream(seed=seed)
    assert s._state.sparse is active
    out = []
    for rows, rate, ms in schedule:
        s.set_stimulus(rows, rate) if rows is not None else s.set_stimulus(None, 0.0)
        c = s.advance(ms)
        d = np.zeros(gpu.n, dtype=np.int64)
        d[c.active_rows] = c.counts
        out.append(d)
    s.close()
    return np.stack(out)


def test_active_set_stream_equals_the_dense_kernel_across_stimulus_changes():
    cpu = ring_net()
    schedule = [(None, 0.0, 20.0), ([0, 1, 2, 3, 4, 5, 6, 7], 800.0, 100.0), ([10, 11, 12, 13], 600.0, 60.0),
                (None, 0.0, 150.0), ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1000.0, 80.0)]
    dense = stream_counts(cpu, False, schedule)
    sparse = stream_counts(cpu, True, schedule)
    assert dense.sum() > 0
    assert np.array_equal(dense, sparse)


def test_active_set_run_equals_the_dense_kernel():
    cpu = ring_net(seed=5)
    res = []
    for active in (False, True):
        gpu = gs.GPUSimulation.from_simulation(cpu)
        gpu.active_set = active
        res.append(gpu.run(list(range(8)), 900.0, duration_ms=300, trials=1, seed=2).counts)
    assert res[0].sum() > 0 and np.array_equal(res[0], res[1])


def test_multi_trial_runs_keep_the_dense_kernel():
    gpu = gs.GPUSimulation.from_simulation(ring_net())
    r = gpu.run([0, 1, 2], 500.0, duration_ms=50, trials=2, seed=1)
    assert r.trials == 2


def test_switching_between_plain_and_active_set_kernels_is_exact(monkeypatch):
    """Thresholds forced so that every busy chunk runs on the plain kernels and every quiet one rebuilds the work lists
    from the state: the stream must still equal the dense one tick for tick."""
    cpu = ring_net(seed=9)
    schedule = [(None, 0.0, 10.0), ([0, 1, 2, 3, 4, 5], 900.0, 30.0), ([0, 1, 2, 3, 4, 5], 900.0, 20.0),
                (None, 0.0, 30.0), (None, 0.0, 200.0), ([20, 21, 22, 23, 24, 25, 26, 27], 1000.0, 20.0),
                (None, 0.0, 40.0), (None, 0.0, 300.0)] * 2
    dense = stream_counts(cpu, False, schedule)
    monkeypatch.setattr(gs, "SWITCH_TO_DENSE", 10**6)
    monkeypatch.setattr(gs, "SWITCH_TO_SPARSE", 10**6)
    modes = []
    real = gs._GPUState._choose_mode

    def spy(self, first, p_dev):
        out = real(self, first, p_dev)
        modes.append(out)
        return out

    monkeypatch.setattr(gs._GPUState, "_choose_mode", spy)
    mixed = stream_counts(cpu, True, schedule)
    assert dense.sum() > 0 and True in modes and False in modes
    assert np.array_equal(dense, mixed)
