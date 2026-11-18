"""GPU simulator tests: skipped without CuPy and a CUDA device. Short runs only (VRAM is shared).

The real-connectome tests compare the GPU against the CPU reference statistically: the RNG
streams differ, so counts are never bit-identical.
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy", reason="CuPy (the `gpu` extra) is not installed")

from flybrain import gpu_simulation as gs  # noqa: E402
from flybrain.simulation import LIFParams, Simulation, SimulationError, StimulusSchedule  # noqa: E402

try:
    HAVE_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # noqa: BLE001
    HAVE_GPU = False
pytestmark = pytest.mark.skipif(not HAVE_GPU, reason="no CUDA device")

CHAIN = [(62, 61, 1000), (61, 60, 1000), (62, 5, 1000), (60, 0, 1000)]


def pair(edges=CHAIN, n=63):
    pre, post, w = (np.array(c) for c in zip(*edges))
    cpu = Simulation.from_arrays(pre, post, w, n_neurons=n)
    return cpu, gs.GPUSimulation.from_simulation(cpu)


@pytest.fixture(autouse=True)
def _free_pool():
    yield
    cp.get_default_memory_pool().free_all_blocks()


def test_same_api_and_fingerprint_as_the_cpu():
    cpu, gpu = pair()
    assert gpu.n == cpu.n and gpu.params == cpu.params and gpu.fingerprint == cpu.fingerprint
    assert gpu.backend == "gpu" and cpu.backend == "cpu"


def test_result_shapes_and_dtypes_match_the_cpu():
    cpu, gpu = pair()
    a = cpu.run([62], 300.0, duration_ms=100, trials=3, bin_ms=10.0)
    b = gpu.run([62], 300.0, duration_ms=100, trials=3, bin_ms=10.0)
    for f in ("counts", "rate_hz", "trial_counts", "active_rows", "frames"):
        assert getattr(a, f).dtype == getattr(b, f).dtype and getattr(a, f).ndim == getattr(b, f).ndim, f
    assert b.counts.shape == a.counts.shape and b.trial_counts.shape == (3, 63)
    assert b.frames.shape[1] == 10 and b.trials == 3 and b.bin_ms == 10.0


def test_the_drive_propagates_down_the_chain():
    _, gpu = pair()
    res = gpu.run([62], 300.0, duration_ms=200, trials=2)
    assert res.counts[62] > 0 and res.counts[61] > 0 and res.counts[60] > 0 and res.counts[0] > 0
    assert res.counts[1] == 0  # unconnected neuron stays silent


def test_the_same_seed_gives_the_same_stream():
    _, gpu = pair()
    a = gpu.run([62], 300.0, duration_ms=100, trials=2, seed=7)
    b = gpu.run([62], 300.0, duration_ms=100, trials=2, seed=7)
    c = gpu.run([62], 300.0, duration_ms=100, trials=2, seed=8)
    assert np.array_equal(a.trial_counts[:, 62], b.trial_counts[:, 62])  # the Poisson target counts are exact
    assert not np.array_equal(a.trial_counts[:, 62], c.trial_counts[:, 62])


def test_trials_are_independent_streams():
    _, gpu = pair()
    res = gpu.run([62], 100.0, duration_ms=400, trials=4)
    assert len(set(res.trial_counts[:, 62].tolist())) > 1


def test_binned_frames_sum_to_the_counts():
    _, gpu = pair()
    res = gpu.run([62], 300.0, duration_ms=100, trials=2, bin_ms=10.0)
    assert np.array_equal(res.frames.sum(axis=1), res.counts[res.active_rows])
    assert np.array_equal(res.active_rows, np.flatnonzero(res.counts))


def test_no_spikes_gives_empty_frames():
    _, gpu = pair()
    res = gpu.run([62], 0.0, duration_ms=50, trials=1, bin_ms=10.0)
    assert res.counts.sum() == 0 and res.frames.shape == (0, 5) and len(res.active_rows) == 0


def test_silencing_cuts_the_outgoing_synapses_only():
    _, gpu = pair()
    res = gpu.run([62], 300.0, duration_ms=200, trials=2, silence_rows=[61])
    assert res.counts[61] > 0 and res.counts[60] == 0 and res.counts[0] == 0  # 61 still spikes, 60 gets nothing
    assert res.counts[5] > 0  # the other branch is untouched


def test_invalid_inputs_raise_the_same_errors():
    _, gpu = pair()
    with pytest.raises(SimulationError):
        gpu.run([62], -1.0, duration_ms=100)
    with pytest.raises(SimulationError):
        gpu.run([999], 100.0, duration_ms=100)
    with pytest.raises(SimulationError):
        gpu.run([62], 100.0, duration_ms=100, trials=0)


def test_run_schedule_segments_open_and_close():
    _, gpu = pair()
    sched = StimulusSchedule([(50.0, 100.0, [62], 400.0)])
    res = gpu.run_schedule(sched, duration_ms=150, trials=2, bin_ms=10.0)
    row = res.frames[list(res.active_rows).index(62)]
    assert row[:5].sum() == 0 and row[5:10].sum() > 0 and row[10:].sum() == 0  # driven only in 50..100 ms


def test_run_schedule_overlapping_rates_add_up():
    _, gpu = pair()
    one = gpu.run_schedule(StimulusSchedule([(0.0, 400.0, [62], 200.0)]), duration_ms=400, trials=4)
    two = gpu.run_schedule(StimulusSchedule([(0.0, 400.0, [62], 100.0), (0.0, 400.0, [62], 100.0)]),
                           duration_ms=400, trials=4, seed=1)
    ref = gpu.run([62], 200.0, duration_ms=400, trials=4, seed=2)
    assert abs(two.counts[62] - ref.counts[62]) < 0.3 * ref.counts[62]
    assert abs(one.counts[62] - ref.counts[62]) < 0.3 * ref.counts[62]


def test_run_schedule_validates_like_the_cpu():
    _, gpu = pair()
    with pytest.raises(SimulationError):
        gpu.run_schedule(StimulusSchedule([(0.0, 500.0, [62], 100.0)]), duration_ms=100)


def test_a_spike_buffer_overflow_raises_instead_of_dropping():
    cpu, _ = pair()
    gpu = gs.GPUSimulation.from_simulation(cpu, max_spikes_per_step=1)
    with pytest.raises(gs.GPUOverflowError):
        gpu.run([62, 61, 60, 5, 0, 1, 2, 3], 5000.0, duration_ms=100, trials=2)


def test_the_pool_limit_is_set_and_the_pool_is_released():
    _, gpu = pair()
    gpu.run([62], 100.0, duration_ms=50)
    pool = cp.get_default_memory_pool()
    assert pool.get_limit() > 0 and pool.get_limit() <= gs.POOL_LIMIT_MB * 2**20
    assert pool.total_bytes() <= 64 * 2**20  # blocks were returned to the driver after the run


def test_a_run_over_the_pool_limit_is_a_capacity_error():
    cpu, _ = pair()
    gpu = gs.GPUSimulation.from_simulation(cpu, pool_limit_mb=1)
    with pytest.raises(gs.GPUCapacityError):
        gpu.run([62], 100.0, duration_ms=50, trials=10)


# -- the real connectome (needs data/processed/sim_weights_783.npz and the annotations) ---------

from pathlib import Path  # noqa: E402

from flybrain.sim_service import DEFAULT_ANNOTATIONS, DEFAULT_WEIGHTS  # noqa: E402

REAL = Path(DEFAULT_WEIGHTS).is_file() and Path(DEFAULT_ANNOTATIONS).is_file()
real = pytest.mark.skipif(not REAL, reason="real simulation weights not available")


@pytest.fixture(scope="module")
def brain():
    from scipy import sparse

    from flybrain.atlas import RowIndex, load_atlas
    from flybrain.tools import load_presets

    z = np.load(DEFAULT_WEIGHTS)
    n = len(z["indptr"]) - 1
    csr = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(n, n))
    atlas = load_atlas(Path(DEFAULT_ANNOTATIONS))
    idx, presets = RowIndex(atlas), load_presets()

    def rows(name):
        r = idx.positions(presets[name]["root_ids"])
        return r[r >= 0]

    cpu = Simulation(csr, LIFParams())
    yield cpu, gs.GPUSimulation.from_simulation(cpu), rows
    cp.get_default_memory_pool().free_all_blocks()


@real
def test_sugar_drives_mn9_like_the_cpu(brain):
    cpu, gpu, rows = brain
    sugar, mn9 = rows("sugar"), rows("mn9")
    a = cpu.run(sugar, 100.0, duration_ms=1000, trials=10, seed=0)
    b = gpu.run(sugar, 100.0, duration_ms=1000, trials=10, seed=0)
    ca, cb = a.group_trial_hz(mn9), b.group_trial_hz(mn9)
    half = 2.576 * ca.std(ddof=1) / np.sqrt(len(ca))  # the CPU's 99% CI of the mean
    assert 40 < ca.mean() < 90  # the published ~63 Hz regime
    assert abs(cb.mean() - ca.mean()) < max(half, 0.1 * ca.mean())


@real
def test_sugar_plus_bitter_keeps_mn9_low(brain):
    cpu, gpu, rows = brain
    mn9 = rows("mn9")
    both = np.union1d(rows("sugar"), rows("bitter"))
    # CPU reference: sugar+bitter on the same neuron set is not the published inhibition test,
    # so compare to the CPU on the identical drive instead of a fixed number
    sugar, bitter = rows("sugar"), rows("bitter")
    sched = StimulusSchedule([(0.0, 1000.0, sugar, 100.0), (0.0, 1000.0, bitter, 100.0)])
    a = cpu.run_schedule(sched, duration_ms=1000, trials=6, seed=0)
    b = gpu.run_schedule(sched, duration_ms=1000, trials=6, seed=0)
    ca, cb = a.group_trial_hz(mn9), b.group_trial_hz(mn9)
    base = cpu.run(sugar, 100.0, duration_ms=1000, trials=6, seed=0).group_trial_hz(mn9).mean()
    assert both.size > 0 and cb.mean() < 0.6 * base  # bitter suppresses the sugar response on the GPU too
    assert abs(cb.mean() - ca.mean()) < max(5.0, 0.25 * max(ca.mean(), 1.0))


@real
def test_active_count_and_per_neuron_rates_agree(brain):
    cpu, gpu, rows = brain
    sugar = rows("sugar")
    a = cpu.run(sugar, 100.0, duration_ms=1000, trials=10, seed=3)
    b = gpu.run(sugar, 100.0, duration_ms=1000, trials=10, seed=4)
    act_a, act_b = np.flatnonzero(a.rate_hz > 1.0), np.flatnonzero(b.rate_hz > 1.0)
    assert abs(len(act_a) - len(act_b)) < 0.05 * len(act_a)
    union = np.union1d(act_a, act_b)
    corr = np.corrcoef(a.rate_hz[union], b.rate_hz[union])[0, 1]
    assert corr > 0.99
