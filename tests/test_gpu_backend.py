"""Backend selection and GPU health, with fakes: no CuPy and no GPU needed."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from flybrain import gpu_simulation as gs
from flybrain.api import create_app
from flybrain.gpu_simulation import GPUCapacityError, GPUOverflowError, GPUProbe, estimate_run_bytes
from flybrain.sim_service import SimulationService, backend_mode
from flybrain.simulation import Simulation, SimulationError

EDGES = [(62, 61, 1000), (61, 60, 1000), (62, 5, 1000), (60, 0, 1000)]
USABLE = GPUProbe(True, "fake GPU, 3000 MiB free", name="Fake GPU", free_mb=3000.0, total_mb=4096.0)


def tiny_sim(n=63) -> Simulation:
    pre, post, w = (np.array(c) for c in zip(*EDGES))
    return Simulation.from_arrays(pre, post, w, n_neurons=n)


class FakeGPU(Simulation):
    """A 'GPU' simulation backed by the CPU code; `fail` is raised by every run."""

    backend = "gpu"
    fail: Exception | None = None

    @classmethod
    def of(cls, sim, fail=None):
        out = cls(sim._csr(), sim.params)
        out.fail = fail
        return out

    def _execute(self, *args, **kw):
        if self.fail is not None:
            raise self.fail
        return super()._execute(*args, **kw)


def service(mode="auto", probe=USABLE, fail=None, factory_error=None):
    calls = {"probe": 0, "factory": 0}

    def gpu_probe():
        calls["probe"] += 1
        return probe

    def gpu_factory(sim):
        calls["factory"] += 1
        if factory_error is not None:
            raise factory_error
        return FakeGPU.of(sim, fail)

    svc = SimulationService.from_simulation(tiny_sim(), backend=mode, gpu_probe=gpu_probe, gpu_factory=gpu_factory)
    return svc, calls


def run(svc, **kw):
    kw.setdefault("duration_ms", 100)
    kw.setdefault("trials", 2)
    with svc.exclusive():
        return svc.run([62], 100.0, **kw)


# -- configuration -------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [(None, "auto"), ("", "auto"), ("CPU", "cpu"), (" gpu ", "gpu"), ("auto", "auto")])
def test_backend_mode_reads_the_environment(monkeypatch, raw, expected):
    monkeypatch.delenv("FLYBRAIN_SIM_BACKEND", raising=False)
    if raw is not None:
        monkeypatch.setenv("FLYBRAIN_SIM_BACKEND", raw)
    assert backend_mode() == expected


def test_backend_mode_ignores_an_unknown_value_with_a_warning(monkeypatch, caplog):
    monkeypatch.setenv("FLYBRAIN_SIM_BACKEND", "tpu")
    with caplog.at_level(logging.WARNING):
        assert backend_mode() == "auto"
    assert "tpu" in caplog.text


# -- selection ----------------------------------------------------------------------------

def test_cpu_mode_never_probes_the_gpu():
    svc, calls = service("cpu")
    run(svc)
    assert (svc.backend, calls) == ("cpu", {"probe": 0, "factory": 0})
    assert "FLYBRAIN_SIM_BACKEND=cpu" in svc.backend_reason


def test_backend_is_pending_before_the_first_run():
    svc, calls = service("auto")
    assert svc.backend == "pending" and calls["probe"] == 0


@pytest.mark.parametrize("reason", ["cupy is not installed", "no CUDA device", "only 300 MiB VRAM free (< 1536 MiB)"])
def test_auto_falls_back_to_cpu_when_the_probe_says_no(reason):
    svc, calls = service("auto", probe=GPUProbe(False, reason))
    run(svc)
    assert (svc.backend, calls["factory"]) == ("cpu", 0)
    assert reason in svc.backend_reason


def test_gpu_mode_also_respects_the_probe():
    svc, calls = service("gpu", probe=GPUProbe(False, "only 300 MiB VRAM free"))
    run(svc)
    assert svc.backend == "cpu" and "gpu requested" in svc.backend_reason and calls["factory"] == 0


def test_auto_uses_the_gpu_when_usable():
    svc, calls = service("auto")
    res, hit = run(svc, bin_ms=10.0)
    assert (svc.backend, calls, hit) == ("gpu", {"probe": 1, "factory": 1}, False)
    assert res.counts.sum() > 0
    run(svc, seed=1)
    assert calls == {"probe": 1, "factory": 1}  # probed and built once


def test_a_factory_failure_falls_back_and_is_not_retried(caplog):
    svc, calls = service("auto", factory_error=RuntimeError("CUDA out of memory"))
    with caplog.at_level(logging.WARNING):
        res, _ = run(svc)
        run(svc, seed=2)
    assert svc.backend == "cpu" and calls["factory"] == 1
    assert "CUDA out of memory" in svc.backend_reason and "CUDA out of memory" in caplog.text
    cpu, _ = run(service("cpu")[0])
    assert np.array_equal(res.counts, cpu.counts)  # the fallback is the CPU result


def test_a_run_failure_reruns_on_cpu_and_disables_the_gpu():
    svc, calls = service("auto", fail=RuntimeError("illegal memory access"))
    res, _ = run(svc)
    assert svc.backend == "cpu" and "illegal memory access" in svc.backend_reason
    cpu, _ = run(service("cpu")[0])
    assert np.array_equal(res.counts, cpu.counts)
    run(svc, seed=3)
    assert calls["factory"] == 1


@pytest.mark.parametrize("error", [GPUOverflowError("spike buffer overflow"), GPUCapacityError("needs 2 GiB")])
def test_overflow_and_capacity_rerun_that_run_on_cpu_but_keep_the_gpu(error):
    svc, _ = service("auto", fail=error)
    res, _ = run(svc)
    cpu, _ = run(service("cpu")[0])
    assert np.array_equal(res.counts, cpu.counts)
    assert svc.backend == "gpu"


def test_input_errors_propagate_and_do_not_disable_the_gpu():
    svc, _ = service("auto")
    with svc.exclusive(), pytest.raises(SimulationError):
        svc.run([62], -5.0, duration_ms=100)
    assert svc.backend == "gpu"


def test_run_schedule_falls_back_too():
    from flybrain.simulation import StimulusSchedule

    svc, _ = service("auto", fail=RuntimeError("boom"))
    sched = StimulusSchedule([(0, 50, [62], 100.0)])
    with svc.exclusive():
        res, _ = svc.run_schedule(sched, duration_ms=50, trials=1)
    assert res.counts.shape == (63,) and svc.backend == "cpu"


def test_the_cache_key_includes_the_backend():
    svc, _ = service("auto")
    _, first = run(svc)
    _, again = run(svc)
    assert (first, again) == (False, True)
    svc._disable_gpu("test")
    _, after = run(svc)
    assert after is False  # RNG streams differ per backend: never share an entry


# -- probe --------------------------------------------------------------------------------

def fake_cupy(devices=1, free=3000, total=4096, name=b"Fake GPU"):
    runtime = SimpleNamespace(
        getDeviceCount=lambda: devices,
        memGetInfo=lambda: (free * 2**20, total * 2**20),
        getDeviceProperties=lambda i: {"name": name},
    )
    return SimpleNamespace(cuda=SimpleNamespace(runtime=runtime))


def test_probe_without_cupy():
    def missing():
        raise ImportError("No module named 'cupy'")

    p = gs.probe_gpu(loader=missing)
    assert not p.usable and "cupy" in p.reason


def test_probe_without_a_device():
    p = gs.probe_gpu(loader=lambda: fake_cupy(devices=0))
    assert not p.usable and "device" in p.reason


def test_probe_with_low_vram():
    p = gs.probe_gpu(min_free_mb=1536, loader=lambda: fake_cupy(free=900))
    assert not p.usable and "900" in p.reason and "1536" in p.reason


def test_probe_ok():
    p = gs.probe_gpu(min_free_mb=1536, loader=lambda: fake_cupy(free=3000))
    assert p.usable and p.name == "Fake GPU" and p.free_mb == 3000


def test_probe_survives_a_cuda_error():
    cp = fake_cupy()
    cp.cuda.runtime.memGetInfo = lambda: (_ for _ in ()).throw(RuntimeError("driver mismatch"))
    p = gs.probe_gpu(loader=lambda: cp)
    assert not p.usable and "driver mismatch" in p.reason


def test_the_margin_is_documented_and_sane():
    assert gs.MIN_FREE_VRAM_MB >= 1024 and gs.POOL_LIMIT_MB < gs.MIN_FREE_VRAM_MB


def test_estimate_grows_with_trials_and_bins():
    base = estimate_run_bytes(139_249, 15_090_883, trials=1, n_bins=0)
    assert estimate_run_bytes(139_249, 15_090_883, trials=10, n_bins=0) > base
    assert estimate_run_bytes(139_249, 15_090_883, trials=1, n_bins=100) > base
    assert base < gs.POOL_LIMIT_MB * 2**20  # one trial of the real brain fits the pool


# -- telemetry and /health/gpu ---------------------------------------------------------------

def test_parse_nvidia_smi():
    info = gs.parse_nvidia_smi("NVIDIA GeForce RTX 3050 Ti Laptop GPU, 52, 26, 4096, 3\n")
    assert info == {"name": "NVIDIA GeForce RTX 3050 Ti Laptop GPU", "temperature_c": 52.0,
                    "vram_used_mb": 26.0, "vram_total_mb": 4096.0, "vram_free_mb": 4070.0, "utilization_pct": 3.0}


@pytest.mark.parametrize("text", ["", "garbage", "a, b, c"])
def test_parse_nvidia_smi_rejects_junk(text):
    assert gs.parse_nvidia_smi(text) is None


def test_read_telemetry_is_best_effort(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gs.subprocess, "run", boom)
    assert gs.read_telemetry() is None


def test_health_gpu_route(tiny_tsv_path, monkeypatch):
    monkeypatch.setattr("flybrain.api.read_telemetry", lambda: {"name": "Fake", "temperature_c": 50.0,
                                                                  "vram_used_mb": 100.0, "vram_free_mb": 3900.0})
    svc, _ = service("auto")
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc)) as client:
        before = client.get("/health/gpu").json()
        run(svc)
        after = client.get("/health/gpu").json()
        assert client.get("/health").json()["simulation"] is True  # /health keeps its shape
    assert before["backend"] == "pending" and before["mode"] == "auto"
    assert after["backend"] == "gpu" and after["reason"] and after["gpu"]["temperature_c"] == 50.0


def test_health_gpu_route_without_telemetry(tiny_tsv_path, monkeypatch):
    monkeypatch.setattr("flybrain.api.read_telemetry", lambda: None)
    with TestClient(create_app(atlas_path=tiny_tsv_path)) as client:
        body = client.get("/health/gpu").json()
    assert body["gpu"] is None and body["backend"] in {"pending", "cpu"}
