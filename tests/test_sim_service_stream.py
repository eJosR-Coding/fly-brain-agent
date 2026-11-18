"""SimulationService: live streams follow the backend decision and the thermal guard (fakes, no GPU needed)."""

import numpy as np

from flybrain.gpu_simulation import GPUCapacityError, ThermalGuard
from flybrain.sim_service import SimulationService
from test_gpu_backend import USABLE, FakeGPU, tiny_sim


class GPUWithStream(FakeGPU):
    stream_error = None

    @classmethod
    def of(cls, sim, fail=None):
        return super().of(sim, fail)

    def open_stream(self, seed=0, silenced=None):
        if self.stream_error is not None:
            raise self.stream_error
        s = super().open_stream(seed, silenced)
        s.backend = "gpu"
        return s


def hot_guard(temp):
    return ThermalGuard(read=lambda: {"temperature_c": temp}, interval_s=0.0)


def make(mode="auto", temp=None, stream_error=None, probe=USABLE):
    def factory(sim):
        g = GPUWithStream.of(sim)
        g.stream_error = stream_error
        return g

    return SimulationService.from_simulation(
        tiny_sim(), backend=mode, gpu_probe=lambda: probe, gpu_factory=factory,
        thermal=None if temp is None else hot_guard(temp),
    )


def test_stream_uses_the_gpu_when_decided_and_cool():
    svc = make(temp=60.0)
    assert svc.open_stream(seed=1).backend == "gpu"


def test_stream_uses_the_cpu_in_cpu_mode():
    assert make(mode="cpu").open_stream().backend == "cpu"


def test_hot_gpu_gets_no_new_stream_and_the_reason_says_so():
    svc = make(temp=85.0)
    assert svc.open_stream().backend == "cpu"
    assert "hot" in svc.stream_note.lower() and "85" in svc.stream_note
    assert svc.backend == "gpu"  # the GPU stays enabled: it is a pause, not a failure


def test_stream_capacity_error_falls_back_to_the_cpu_for_that_session_only():
    svc = make(temp=60.0, stream_error=GPUCapacityError("no room"))
    assert svc.open_stream().backend == "cpu" and svc.backend == "gpu"


def test_unexpected_gpu_stream_error_disables_the_gpu():
    svc = make(temp=60.0, stream_error=RuntimeError("driver died"))
    assert svc.open_stream().backend == "cpu" and svc.backend == "cpu"


def test_hot_gpu_runs_the_one_shot_simulation_on_the_cpu_without_disabling_the_gpu():
    svc = make(temp=85.0)
    with svc.exclusive():
        res, hit = svc.run([62], 100.0, duration_ms=100, trials=1)
    assert not hit and res.counts.sum() > 0 and svc.backend == "gpu"
    # the cache key says CPU: a later cool run must not reuse it as a GPU result
    svc.thermal = hot_guard(60.0)
    with svc.exclusive():
        _, hit2 = svc.run([62], 100.0, duration_ms=100, trials=1)
    assert not hit2


def test_thermal_status_is_exposed_when_a_guard_is_set():
    svc = make(temp=85.0)
    svc.thermal.allow()
    assert svc.thermal_status()["hot"] is True
    assert make().thermal_status() is None


def test_chunk_counts_are_plain_numpy():
    s = make(mode="cpu").open_stream()
    s.set_stimulus([62], 300.0)
    c = s.advance(50.0)
    assert isinstance(c.active_rows, np.ndarray)
