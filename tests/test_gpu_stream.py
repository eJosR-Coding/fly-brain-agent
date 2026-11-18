"""GPU streaming brain: skipped without CuPy and a CUDA device. Tiny nets, short runs (VRAM is shared)."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy", reason="CuPy (the `gpu` extra) is not installed")

from flybrain import gpu_simulation as gs  # noqa: E402
from flybrain.simulation import Simulation, SimulationError  # noqa: E402

try:
    HAVE_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # noqa: BLE001
    HAVE_GPU = False
pytestmark = pytest.mark.skipif(not HAVE_GPU, reason="no CUDA device")

CHAIN = [(62, 61, 1000), (61, 60, 1000), (62, 5, 1000), (60, 0, 1000)]


def pair(n=63):
    pre, post, w = (np.array(c) for c in zip(*CHAIN))
    cpu = Simulation.from_arrays(pre, post, w, n_neurons=n)
    return cpu, gs.GPUSimulation.from_simulation(cpu)


@pytest.fixture(autouse=True)
def _free_pool():
    yield
    cp.get_default_memory_pool().free_all_blocks()


def dense(chunk, n):
    out = np.zeros(n, dtype=np.int64)
    out[chunk.active_rows] = chunk.counts
    return out


def test_gpu_stream_chunks_are_bit_identical_to_one_gpu_run():
    _, gpu = pair()
    ref = gpu.run([62], 300.0, duration_ms=500, trials=1, seed=4)
    s = gpu.open_stream(seed=4)
    assert s.backend == "gpu"
    s.set_stimulus([62], 300.0)
    total = sum(dense(s.advance(50.0), gpu.n) for _ in range(10))
    assert np.array_equal(total, ref.counts) and ref.counts.sum() > 0
    assert s.t_ms == pytest.approx(500.0)
    s.close()


def test_gpu_stream_stimulus_changes_and_validation():
    _, gpu = pair()
    s = gpu.open_stream(seed=1)
    assert len(s.advance(50.0).active_rows) == 0
    s.set_stimulus([62], 400.0)
    assert 62 in s.advance(100.0).active_rows
    with pytest.raises(SimulationError):
        s.set_stimulus([999], 10.0)
    s.set_stimulus(None, 0.0)
    s.advance(100.0)
    assert 62 not in s.advance(100.0).active_rows
    s.close()


def test_gpu_stream_agrees_with_the_cpu_stream_statistically():
    cpu, gpu = pair()
    rates = []
    for sim in (cpu, gpu):
        tot = 0
        for seed in range(6):
            st = sim.open_stream(seed=seed)
            st.set_stimulus([62], 300.0)
            tot += sum(st.advance(50.0).group_hz([60, 61]) for _ in range(10)) / 10
            st.close()
        rates.append(tot / 6)
    assert rates[0] > 0 and abs(rates[0] - rates[1]) < 0.25 * rates[0] + 5
