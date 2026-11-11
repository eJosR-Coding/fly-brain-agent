"""Dose-response over a tiny network (no real weights, no LLM)."""

import asyncio
import json

import numpy as np
import pytest

from flybrain import experiments, tools
from flybrain.selections import SelectionStore
from flybrain.sim_service import SimulationService
from flybrain.simulation import Simulation

# tiny_atlas rows: 0..59 PEN_a, 60 EPG, 61 T4a (optic), 62 ORN_DA1 (sensory)
# ORN -> T4a -> EPG is the pathway; ORN -> PEN row 5 and EPG -> PEN row 0 are branches that cannot reach EPG.
EDGES = [(62, 61, 1000), (61, 60, 1000), (62, 5, 1000), (60, 0, 1000)]


@pytest.fixture
def sims(tiny_atlas):
    pre, post, w = (np.array(c) for c in zip(*EDGES))
    return SimulationService.from_simulation(Simulation.from_arrays(pre, post, w, n_neurons=len(tiny_atlas.neurons)))


@pytest.fixture
def store():
    return SelectionStore()


def ticket(tiny_atlas, store, **filters):
    return tools.search_neurons(tiny_atlas, store, **filters)["selection_id"]


@pytest.fixture
def orn(tiny_atlas, store):
    return ticket(tiny_atlas, store, cell_type="ORN_DA1")


@pytest.fixture
def epg(tiny_atlas, store):
    return ticket(tiny_atlas, store, cell_type="EPG")


# --- half-max rate -----------------------------------------------------------------------


def test_half_max_lands_exactly_on_a_point_when_the_curve_passes_through_it():
    assert experiments.half_max_rate([10, 25, 50, 100, 200], [0, 10, 50, 90, 100]) == 50


def test_half_max_is_interpolated_linearly_between_the_bracketing_rates():
    assert experiments.half_max_rate([10, 50, 100], [10, 60, 100]) == pytest.approx(42.0)
    assert experiments.half_max_rate([10, 20, 40], [0, 20, 40]) == pytest.approx(20.0)


def test_half_max_is_none_without_a_response_or_when_already_above_half_at_the_lowest_rate():
    assert experiments.half_max_rate([10, 50], [0, 0]) is None
    assert experiments.half_max_rate([10, 50], [90, 100]) is None  # half-max lies below the tested range


# --- dose_response -----------------------------------------------------------------------


def dose(tiny_atlas, store, sims, orn, epg, **kw):
    return experiments.dose_response(
        tiny_atlas, store, sims, orn, epg, **{"rates_hz": [10, 50, 300], "duration_ms": 100, "trials": 3, **kw}
    )


def test_dose_response_returns_a_point_per_rate_with_intervals(tiny_atlas, store, sims, orn, epg):
    out = dose(tiny_atlas, store, sims, orn, epg)
    assert [p["rate_hz"] for p in out["points"]] == [10, 50, 300]
    for p in out["points"]:
        assert set(p) == {"rate_hz", "readout_mean_hz", "ci_low", "ci_high", "active_neurons"}
        assert p["ci_low"] <= p["readout_mean_hz"] <= p["ci_high"]
    means = [p["readout_mean_hz"] for p in out["points"]]
    assert means == sorted(means) and means[-1] > means[0]  # more input, more output
    assert out["runtime_s"] >= 0 and "model prediction" in out["note"]
    assert out["stimulus"] == {"neurons": 1, "duration_ms": 100.0, "trials": 3}
    assert len(json.dumps(out)) < 1800


def test_dose_response_reports_the_ec50_of_its_own_curve(tiny_atlas, store, sims, orn, epg):
    out = dose(tiny_atlas, store, sims, orn, epg, rates_hz=[10, 50, 100, 300])
    curve = [(p["rate_hz"], p["readout_mean_hz"]) for p in out["points"]]
    assert out["ec50_hz"] == pytest.approx(experiments.half_max_rate(*zip(*curve)), abs=0.05)
    assert out["ec50_hz"] is None or 10 <= out["ec50_hz"] <= 300


def test_a_dead_readout_has_no_ec50_and_says_why(tiny_atlas, store, sims, orn, epg):
    out = dose(tiny_atlas, store, sims, epg, orn, rates_hz=[10, 300], trials=2)  # nothing projects onto the ORN
    assert [p["readout_mean_hz"] for p in out["points"]] == [0, 0]
    assert out["ec50_hz"] is None and "no response" in out["ec50_note"]


def test_dose_response_sorts_dedupes_and_caps_the_inputs_with_notes(tiny_atlas, store, sims, orn, epg):
    out = dose(tiny_atlas, store, sims, orn, epg, rates_hz=[300, 50, 50, 5000, 0.2], duration_ms=5, trials=99)
    assert [p["rate_hz"] for p in out["points"]] == [1, 50, 300]
    assert out["stimulus"]["duration_ms"] == 100.0 and out["stimulus"]["trials"] == 10
    for text in ("rate_hz 0.2 clamped to 1.0", "rate_hz capped to 300.0", "trials capped to 10"):
        assert text in out["note"]
    many = dose(tiny_atlas, store, sims, orn, epg, rates_hz=list(range(1, 20)), trials=1, duration_ms=100)
    assert len(many["points"]) == 8 and "at most 8 rates" in many["note"]


@pytest.mark.parametrize("bad", ["fast", [], [True], [1, "x"], None])
def test_dose_response_rejects_bad_rates(tiny_atlas, store, sims, orn, epg, bad):
    with pytest.raises((TypeError, ValueError)):
        experiments.dose_response(tiny_atlas, store, sims, orn, epg, rates_hz=bad)


def test_dose_response_stores_the_curve_for_the_viewer(tiny_atlas, store, sims, orn, epg):
    out = dose(tiny_atlas, store, sims, orn, epg)
    chart = sims.get_chart(out["chart_id"])
    assert chart["kind"] == "dose_response" and chart["id"] == out["chart_id"]
    assert chart["points"] == [
        {"rate_hz": p["rate_hz"], "mean_hz": p["readout_mean_hz"], "ci_low": p["ci_low"], "ci_high": p["ci_high"]}
        for p in out["points"]
    ]
    assert chart["readout_label"] == "EPG" and chart["stimulus_label"] == "ORN_DA1" and chart["ec50_hz"] == out["ec50_hz"]


def test_progress_is_reported_per_rate_and_cleared_afterwards(tiny_atlas, store, sims, orn, epg):
    seen = []
    off = sims.subscribe(lambda p: seen.append(dict(p)))
    dose(tiny_atlas, store, sims, orn, epg)
    off()
    assert [(p["experiment"], p["done"], p["total"]) for p in seen] == [("dose_response", i, 3) for i in range(4)]
    assert sims.progress_snapshot() is None


def test_progress_is_cleared_even_when_the_experiment_fails(tiny_atlas, store, sims, orn):
    with pytest.raises(KeyError):
        experiments.dose_response(tiny_atlas, store, sims, orn, "sel_nope", rates_hz=[10], duration_ms=100, trials=1)
    assert sims.progress_snapshot() is None


# --- over MCP ----------------------------------------------------------------------------


def test_the_server_exposes_dose_response(tiny_atlas, sims):
    from flybrain.mcp_server import build_server

    server = build_server(tiny_atlas, sims=sims)
    by_name = {t.name: t for t in asyncio.run(server.list_tools())}
    assert "dose_response" in by_name and "model" in by_name["dose_response"].description.lower()

    def call(name, args):
        res = asyncio.run(server.call_tool(name, args))
        payload = res[1] if isinstance(res, tuple) else res
        return json.loads(payload[0].text) if isinstance(payload, list) else payload

    orn = call("search_neurons", {"cell_type": "ORN_DA1"})["selection_id"]
    epg = call("search_neurons", {"cell_type": "EPG"})["selection_id"]
    out = call("dose_response", {"selection_id": orn, "readout_selection_id": epg,
                                 "rates_hz": [10, 300], "duration_ms": 100, "trials": 2})
    assert len(out["points"]) == 2 and out["chart_id"].startswith("chart_")
