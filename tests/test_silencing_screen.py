"""Silencing screen over a tiny network (no real weights, no LLM)."""

import asyncio
import json

import numpy as np
import pytest

from flybrain import silencing, tools
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


# --- silencing_screen --------------------------------------------------------------------


def screen(tiny_atlas, store, sims, orn, epg, **kw):
    return silencing.silencing_screen(
        tiny_atlas, store, sims, orn, epg, **{"rate_hz": 300, "duration_ms": 300, "trials": 4, **kw}
    )


def test_screen_ranks_candidates_by_their_effect_on_the_readout(tiny_atlas, store, sims, orn, epg):
    out = screen(tiny_atlas, store, sims, orn, epg)
    rows = out["results"]
    assert [r["label"] for r in rows] == ["T4a", "PEN_a", "PEN_a"]  # T4a feeds EPG, the PEN_a branches do not
    first, second = rows[0], rows[1]
    assert first["delta_hz"] < 0 and first["ci_high"] < 0 and first["significant"] is True and first["p_adj"] < 0.05
    assert second["delta_hz"] == 0 and second["p"] == 1.0 and second["significant"] is False
    assert out["significant_count"] == 1 and out["candidates"] == 3
    assert out["baseline"]["readout_mean_hz"] > 0 and "Holm" in out["note"] and "model prediction" in out["note"]
    assert len(json.dumps(out)) < 2200


def test_screen_never_silences_the_stimulus_or_the_readout(tiny_atlas, store, sims, orn, epg):
    rows = {r["row"] for r in screen(tiny_atlas, store, sims, orn, epg)["results"]}
    assert 62 not in rows and 60 not in rows  # the stimulated ORN and the EPG readout itself


def test_screen_keeps_every_silenced_recording_and_the_baseline(tiny_atlas, store, sims, orn, epg):
    out = screen(tiny_atlas, store, sims, orn, epg)
    base = sims.get(out["baseline_simulation_id"])
    for r in out["results"]:
        rec = sims.get(r["simulation_id"])
        assert rec.compare_id == out["baseline_simulation_id"] and list(rec.silenced_rows) == [r["row"]]
        assert rec.readout["label"] == "EPG" and rec.trials == base.trials
    t4a = next(r for r in out["results"] if r["label"] == "T4a")
    assert max(sims.get(t4a["simulation_id"]).readout["hz"]) == 0 < max(base.readout["hz"])


def test_screen_returns_a_ticket_of_the_candidates(tiny_atlas, store, sims, orn, epg):
    out = screen(tiny_atlas, store, sims, orn, epg)
    assert set(store.get(out["candidates_selection_id"]).root_ids) == {101, 6, 1}  # T4a, PEN rows 5 and 0 (roots 6 and 1)


def test_screen_accepts_a_candidate_ticket(tiny_atlas, store, sims, orn, epg):
    only_t4a = ticket(tiny_atlas, store, cell_type="T4a")
    out = screen(tiny_atlas, store, sims, orn, epg, candidates=only_t4a)
    assert [r["label"] for r in out["results"]] == ["T4a"] and out["significant_count"] == 1


def test_screen_rejects_an_unknown_candidates_mode(tiny_atlas, store, sims, orn, epg):
    with pytest.raises(ValueError, match="top_non_sensory"):
        screen(tiny_atlas, store, sims, orn, epg, candidates="everything")


def test_screen_with_no_active_candidates_says_so(tiny_atlas, store, sims, epg):
    quiet = ticket(tiny_atlas, store, cell_type="T4a")  # stimulating T4a: only EPG and PEN row 0 follow
    out = screen(tiny_atlas, store, sims, quiet, epg, candidates=ticket(tiny_atlas, store, cell_type="ORN_DA1"))
    assert out["results"] == [] and "hint" in out


def test_screen_caps_n_and_reports_progress_per_candidate(tiny_atlas, store, sims, orn, epg):
    seen = []
    off = sims.subscribe(lambda p: seen.append((p["experiment"], p["done"], p["total"])))
    out = screen(tiny_atlas, store, sims, orn, epg, n=50)
    off()
    assert "n capped to 12" in out["note"]
    assert seen == [("silencing_screen", i, 3) for i in range(4)]  # only 3 candidates exist, so 3 steps


def test_screen_chart_for_the_viewer(tiny_atlas, store, sims, orn, epg):
    out = screen(tiny_atlas, store, sims, orn, epg)
    chart = sims.get_chart(out["chart_id"])
    assert chart["kind"] == "screen" and chart["readout_label"] == "EPG"
    assert [r["label"] for r in chart["rows"]] == ["T4a", "PEN_a", "PEN_a"]
    r = chart["rows"][0]
    assert set(r) == {"label", "row", "delta_hz", "ci_low", "ci_high", "p", "significant", "simulation_id"}
    assert r["simulation_id"] == out["results"][0]["simulation_id"]


def test_screen_with_one_trial_cannot_claim_significance(tiny_atlas, store, sims, orn, epg):
    out = screen(tiny_atlas, store, sims, orn, epg, trials=1)
    assert all(r["p"] is None and r["significant"] is False for r in out["results"])
    assert out["significant_count"] == 0 and "at least 2 trials" in out["note"]


# --- over MCP ----------------------------------------------------------------------------


def test_the_server_exposes_the_silencing_screen(tiny_atlas, sims):
    from flybrain.mcp_server import build_server

    server = build_server(tiny_atlas, sims=sims)
    by_name = {t.name: t for t in asyncio.run(server.list_tools())}
    assert "silencing_screen" in by_name and "Holm" in by_name["silencing_screen"].description

    def call(name, args):
        res = asyncio.run(server.call_tool(name, args))
        payload = res[1] if isinstance(res, tuple) else res
        return json.loads(payload[0].text) if isinstance(payload, list) else payload

    orn = call("search_neurons", {"cell_type": "ORN_DA1"})["selection_id"]
    epg = call("search_neurons", {"cell_type": "EPG"})["selection_id"]
    out = call("silencing_screen", {"selection_id": orn, "readout_selection_id": epg, "rate_hz": 300,
                                    "duration_ms": 100, "trials": 2, "n": 2})
    assert len(out["results"]) == 2 and out["chart_id"].startswith("chart_")
