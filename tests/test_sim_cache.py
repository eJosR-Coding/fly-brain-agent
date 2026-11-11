"""The simulation result cache: same experiment twice is instant, keyed by rows + params, never ticket ids."""

import numpy as np
import pytest

from flybrain import experiments, tools
from flybrain.selections import SelectionStore
from flybrain.sim_service import SimulationService
from flybrain.simulation import Simulation

EDGES = [(62, 61, 1000), (61, 60, 1000), (62, 5, 1000), (60, 0, 1000)]


@pytest.fixture
def sims(tiny_atlas):
    pre, post, w = (np.array(c) for c in zip(*EDGES))
    return SimulationService.from_simulation(Simulation.from_arrays(pre, post, w, n_neurons=len(tiny_atlas.neurons)))


@pytest.fixture
def store():
    return SelectionStore()


def run(sims, stim=(62,), rate=100.0, **kw):
    kw.setdefault("duration_ms", 100)
    kw.setdefault("trials", 2)
    kw.setdefault("seed", 0)
    with sims.exclusive():
        return sims.run(list(stim), rate, **kw)


def test_same_run_twice_is_a_hit_and_identical(sims):
    first, hit1 = run(sims, bin_ms=10.0)
    second, hit2 = run(sims, bin_ms=10.0)
    assert (hit1, hit2) == (False, True)
    assert np.array_equal(first.counts, second.counts) and np.array_equal(first.frames, second.frames)


def test_row_order_and_duplicates_do_not_change_the_key(sims):
    _, a = run(sims, stim=(62, 61))
    _, b = run(sims, stim=(61, 62, 62))
    assert (a, b) == (False, True)


@pytest.mark.parametrize("change", [
    {"rate": 50.0}, {"seed": 1}, {"silence_rows": [61]}, {"trials": 3}, {"duration_ms": 200}, {"bin_ms": 10.0},
    {"stim": (61,)},
])
def test_any_parameter_change_is_a_miss(sims, change):
    run(sims)
    _, hit = run(sims, **change)
    assert hit is False


def test_a_different_weight_matrix_never_hits(tiny_atlas):
    n = len(tiny_atlas.neurons)
    a = SimulationService.from_simulation(Simulation.from_arrays([62], [61], [1000], n_neurons=n))
    b = SimulationService.from_simulation(Simulation.from_arrays([62], [61], [-1000], n_neurons=n))
    assert a.simulation().fingerprint != b.simulation().fingerprint


def test_lru_bound_evicts_the_oldest(tiny_atlas):
    pre, post, w = (np.array(c) for c in zip(*EDGES))
    sims = SimulationService.from_simulation(
        Simulation.from_arrays(pre, post, w, n_neurons=len(tiny_atlas.neurons)), max_cached_runs=2)
    run(sims, seed=1)
    run(sims, seed=2)
    run(sims, seed=1)  # touch seed 1 so seed 2 is the oldest
    run(sims, seed=3)  # evicts seed 2
    assert run(sims, seed=1)[1] is True
    assert run(sims, seed=2)[1] is False


def test_the_byte_budget_evicts_too(tiny_atlas):
    pre, post, w = (np.array(c) for c in zip(*EDGES))
    sims = SimulationService.from_simulation(
        Simulation.from_arrays(pre, post, w, n_neurons=len(tiny_atlas.neurons)), max_cache_bytes=1)
    run(sims)
    assert run(sims)[1] is False  # a single entry over budget is not kept


def ticket(tiny_atlas, store, **f):
    return tools.search_neurons(tiny_atlas, store, **f)["selection_id"]


def test_stimulate_second_call_is_cached_even_with_new_ticket_ids(tiny_atlas, store, sims):
    out1 = tools.stimulate(tiny_atlas, store, sims, ticket(tiny_atlas, store, cell_type="ORN_DA1"), rate_hz=100, duration_ms=100)
    out2 = tools.stimulate(tiny_atlas, store, sims, ticket(tiny_atlas, store, cell_type="ORN_DA1"), rate_hz=100, duration_ms=100)
    assert out1["cached"] is False and out2["cached"] is True
    assert out2["active_neurons"] == out1["active_neurons"] and out2["top_responders"] == out1["top_responders"]
    assert "cache" in out2["note"].lower()


def test_stimulate_with_silencing_reuses_the_baseline(tiny_atlas, store, sims):
    orn = ticket(tiny_atlas, store, cell_type="ORN_DA1")
    tools.stimulate(tiny_atlas, store, sims, orn, rate_hz=100, duration_ms=100, trials=3)
    out = tools.stimulate(tiny_atlas, store, sims, orn, rate_hz=100, duration_ms=100, trials=3,
                          silence_selection_id=ticket(tiny_atlas, store, cell_type="T4a"))
    assert out["cached"] is False  # the silenced run is new; only the baseline came from the cache
    assert sims.cache_hits == 1


def test_dose_response_reuses_step_runs(tiny_atlas, store, sims):
    orn = ticket(tiny_atlas, store, cell_type="ORN_DA1")
    epg = ticket(tiny_atlas, store, cell_type="EPG")
    experiments.dose_response(tiny_atlas, store, sims, orn, epg, rates_hz=[10, 100], duration_ms=100, trials=2)
    before = sims.cache_hits
    out = experiments.dose_response(tiny_atlas, store, sims, orn, epg, rates_hz=[10, 100, 200], duration_ms=100, trials=2)
    assert sims.cache_hits == before + 2
    assert out["cached_runs"] == 2
