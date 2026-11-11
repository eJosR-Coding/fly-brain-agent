"""preset_selection and stimulate over a tiny hand-built network (no real weights, no LLM)."""

import asyncio
import json
import threading
import time

import numpy as np
import pytest

from flybrain import tools
from flybrain.selections import SelectionStore
from flybrain.sim_service import SimulationService, SimulationUnavailable, UnknownSimulationError
from flybrain.simulation import Simulation

# tiny_atlas rows: 0..59 PEN_a (root 1..60), 60 EPG (100), 61 T4a (101), 62 ORN_DA1 (102)
EDGES = [(62, 61, 1000), (61, 60, 1000), (60, 0, 1000), (60, 1, -1000)]


@pytest.fixture
def sims(tiny_atlas):
    pre, post, w = (np.array(c) for c in zip(*EDGES))
    return SimulationService.from_simulation(Simulation.from_arrays(pre, post, w, n_neurons=len(tiny_atlas.neurons)))


@pytest.fixture
def store():
    return SelectionStore()


def ticket(tiny_atlas, store, **filters):
    return tools.search_neurons(tiny_atlas, store, **filters)["selection_id"]


PRESETS = {
    "orn": {"description": "olfactory input", "root_ids": [102, 999], "source": "test source"},
}


# --- presets -----------------------------------------------------------------------------


def test_packaged_presets_are_cited_and_complete():
    presets = tools.load_presets()
    assert len(presets["sugar"]["root_ids"]) == 21
    assert presets["mn9"]["root_ids"] == [720575940660219265]
    for p in presets.values():
        assert "91bdd1e" in p["source"] and "MIT" in p["source"] and p["description"]


def test_preset_selection_returns_a_ticket_and_a_source_note(tiny_atlas, store):
    out = tools.preset_selection(tiny_atlas, store, "orn", presets=PRESETS)
    assert out["name"] == "orn" and out["total"] == 1 and out["missing"] == 1
    assert out["source"] == "test source" and out["selection_id"].startswith("sel_")
    assert store.get(out["selection_id"]).root_ids == (102,)
    assert out["by_super_class"] == {"sensory": 1}


def test_unknown_preset_lists_the_available_ones(tiny_atlas, store):
    out = tools.preset_selection(tiny_atlas, store, "nope", presets=PRESETS)
    assert out["selection_id"] is None and out["available"] == ["orn"]
    assert "orn" in out["hint"]


def test_preset_with_no_neuron_in_the_atlas_is_an_error_message(tiny_atlas, store):
    out = tools.preset_selection(tiny_atlas, store, "sugar")  # real ids are not in the tiny atlas
    assert out["selection_id"] is None and out["total"] == 0 and "None of its 21" in out["hint"]


# --- stimulate ---------------------------------------------------------------------------


def run(tiny_atlas, store, sims, **kw):
    sid = kw.pop("selection_id", None) or ticket(tiny_atlas, store, cell_type="ORN_DA1")
    return tools.stimulate(tiny_atlas, store, sims, sid, **{"rate_hz": 300, "duration_ms": 200, "trials": 2, **kw})


def test_stimulate_returns_a_small_summary_with_tickets(tiny_atlas, store, sims):
    out = run(tiny_atlas, store, sims)
    assert out["active_neurons"] >= 3
    first = out["top_responders"][0]
    assert set(first) == {"root_id", "cell_type", "super_class", "rate_hz"} and first["rate_hz"] > 1
    assert out["top_responders"] == sorted(out["top_responders"], key=lambda r: -r["rate_hz"])
    active_ids = set(store.get(out["selection_id"]).root_ids)
    assert {102, 101, 100} <= active_ids and len(active_ids) == out["active_neurons"]
    assert out["simulation_id"].startswith("sim_") and out["runtime_s"] >= 0
    assert "Shiu et al. 2024" in out["note"] and "not the living fly" in out["note"]
    assert len(json.dumps(out)) < 2000
    assert out["stimulus"] == {"neurons": 1, "rate_hz": 300.0, "duration_ms": 200.0, "trials": 2}


def test_top_non_sensory_skips_sensory_inputs_and_lists_silencing_candidates(tiny_atlas, store, sims):
    out = run(tiny_atlas, store, sims)
    top = out["top_non_sensory"]
    assert 1 <= len(top) <= 5
    assert all(set(r) == {"root_id", "cell_type", "super_class", "rate_hz"} for r in top)
    assert all(r["super_class"] not in ("sensory", "sensory_ascending") for r in top)
    assert 102 not in {r["root_id"] for r in top}  # the stimulated sensory neuron itself
    assert {r["root_id"] for r in top} >= {101, 100}
    assert top == sorted(top, key=lambda r: -r["rate_hz"])
    assert len(json.dumps(out)) < 2000


def test_top_non_sensory_is_empty_when_nothing_fires(tiny_atlas, store, sims):
    out = run(tiny_atlas, store, sims, rate_hz=1, duration_ms=100, trials=1, selection_id=ticket(tiny_atlas, store, cell_type="PEN_a"))
    assert isinstance(out["top_non_sensory"], list)


def test_stimulate_stores_sparse_uint8_frames(tiny_atlas, store, sims):
    out = run(tiny_atlas, store, sims)
    rec = sims.get(out["simulation_id"])
    assert rec.rows.dtype == np.uint32 and rec.frames.dtype == np.uint8
    assert rec.frames.shape == (len(rec.rows), rec.n_bins) and rec.n_bins <= 100
    assert list(rec.rows) == sorted(rec.rows) and 62 in rec.rows
    assert rec.bin_ms * rec.n_bins >= rec.duration_ms and rec.trials == 2


def test_readout_reports_the_group(tiny_atlas, store, sims):
    pen = ticket(tiny_atlas, store, cell_type="PEN_a")
    out = run(tiny_atlas, store, sims, readout_selection_id=pen)
    ro = out["readout"]
    assert ro["neurons"] == 60 and ro["mean_hz"] > 0 and ro["max_hz"] >= ro["mean_hz"]
    assert ro["top"][0]["rate_hz"] == ro["max_hz"] and len(ro["top"]) <= 5


def test_silencing_runs_a_baseline_and_reports_the_change(tiny_atlas, store, sims):
    epg = ticket(tiny_atlas, store, cell_type="EPG")
    pen = ticket(tiny_atlas, store, cell_type="PEN_a")
    out = run(tiny_atlas, store, sims, silence_selection_id=epg, readout_selection_id=pen)
    assert out["silenced_neurons"] == 1
    change = out["readout_change"]
    assert change["baseline_mean_hz"] > 0 and change["silenced_mean_hz"] == 0
    assert change["change_hz"] < 0 and change["change_pct"] == -100.0
    assert out["baseline"]["active_neurons"] > out["active_neurons"]
    assert "2x" in out["note"]


def test_inputs_are_clamped_with_notes(tiny_atlas, store, sims):
    out = run(tiny_atlas, store, sims, rate_hz=5000, duration_ms=5, trials=50)
    assert out["stimulus"] == {"neurons": 1, "rate_hz": 300.0, "duration_ms": 100.0, "trials": 10}
    for text in ("rate_hz capped to 300.0", "duration_ms 5.0 clamped to 100.0", "trials capped to 10"):
        assert text in out["note"]


@pytest.mark.parametrize("kw", [dict(trials=2.5), dict(trials=True), dict(rate_hz="fast"), dict(duration_ms=None)])
def test_bad_types_are_rejected(tiny_atlas, store, sims, kw):
    with pytest.raises(TypeError):
        run(tiny_atlas, store, sims, **kw)


def test_unknown_ticket_and_missing_weights_are_guiding_errors(tiny_atlas, store, sims):
    with pytest.raises(KeyError, match="unknown selection"):
        tools.stimulate(tiny_atlas, store, sims, "sel_nope")
    gone = SimulationService.default(tiny_atlas)  # conftest points the env at nothing
    assert gone.available is False
    with pytest.raises(SimulationUnavailable, match="flybrain-build-sim-weights"):
        run(tiny_atlas, store, gone)


def test_results_are_an_lru_store(tiny_atlas, store):
    svc = SimulationService.from_simulation(Simulation.from_arrays([62], [61], [9], len(tiny_atlas.neurons)), max_results=2)
    ids = [run(tiny_atlas, store, svc, trials=1)["simulation_id"] for _ in range(3)]
    svc.get(ids[1])
    with pytest.raises(UnknownSimulationError, match="expired.*run the simulation again"):
        svc.get(ids[0])
    with pytest.raises(UnknownSimulationError, match="unknown simulation"):
        svc.get("sim_zzzzzz")


def test_only_one_simulation_runs_at_a_time(tiny_atlas, store, sims):
    live, peak = 0, 0
    real = sims.simulation().run

    def spy(*a, **kw):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        time.sleep(0.05)
        try:
            return real(*a, **kw)
        finally:
            live -= 1

    sims.simulation().run = spy
    sid = ticket(tiny_atlas, store, cell_type="ORN_DA1")
    threads = [threading.Thread(target=run, args=(tiny_atlas, store, sims), kwargs={"selection_id": sid, "trials": 1}) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak == 1


# --- over MCP ----------------------------------------------------------------------------


def test_the_server_exposes_both_tools_and_runs_stimulate(tiny_atlas, sims):
    from flybrain.mcp_server import build_server

    server = build_server(tiny_atlas, sims=sims)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"preset_selection", "stimulate"} <= names

    def call(name, args):
        res = asyncio.run(server.call_tool(name, args))
        payload = res[1] if isinstance(res, tuple) else res
        return json.loads(payload[0].text) if isinstance(payload, list) else payload

    sid = call("search_neurons", {"cell_type": "ORN_DA1"})["selection_id"]
    out = call("stimulate", {"selection_id": sid, "rate_hz": 300, "duration_ms": 100, "trials": 1})
    assert out["simulation_id"].startswith("sim_") and out["active_neurons"] >= 1
    assert "simulation" in {t.name: t for t in asyncio.run(server.list_tools())}["stimulate"].description.lower()


# --- comparison recordings (VL4) ---------------------------------------------------------


def test_silencing_keeps_the_baseline_recording_and_links_it(tiny_atlas, store, sims):
    epg = ticket(tiny_atlas, store, cell_type="EPG")
    pen = ticket(tiny_atlas, store, cell_type="PEN_a")
    out = run(tiny_atlas, store, sims, silence_selection_id=epg, readout_selection_id=pen)
    rec = sims.get(out["simulation_id"])
    assert rec.compare_id is not None and rec.compare_id != rec.id
    base = sims.get(rec.compare_id)
    assert base.compare_id is None and base.bin_ms == rec.bin_ms and base.n_bins == rec.n_bins
    assert base.stimulus == rec.stimulus and base.trials == rec.trials
    assert 0 in base.rows and 0 not in rec.rows  # PEN row 0 only fires when EPG is not silenced
    assert list(rec.silenced_rows) == [60] and base.silenced_rows is None


def test_recordings_carry_the_readout_trace_in_hz(tiny_atlas, store, sims):
    epg = ticket(tiny_atlas, store, cell_type="EPG")
    pen = ticket(tiny_atlas, store, cell_type="PEN_a")
    out = run(tiny_atlas, store, sims, silence_selection_id=epg, readout_selection_id=pen)
    rec = sims.get(out["simulation_id"])
    base = sims.get(rec.compare_id)
    assert rec.readout["label"] == "PEN_a" and rec.readout["neurons"] == 60
    assert len(rec.readout["hz"]) == rec.n_bins == len(base.readout["hz"])
    assert max(base.readout["hz"]) > 0 and max(rec.readout["hz"]) == 0
    # the trace is the group's mean rate: averaged over its bins it matches the reported mean
    dur_s = rec.duration_ms / 1000
    mean_hz = sum(base.readout["hz"]) * (rec.bin_ms / 1000) / dur_s
    assert mean_hz == pytest.approx(out["readout_change"]["baseline_mean_hz"], rel=0.05, abs=0.1)


def test_plain_runs_have_no_comparison_fields(tiny_atlas, store, sims):
    rec = sims.get(run(tiny_atlas, store, sims)["simulation_id"])
    assert rec.compare_id is None and rec.silenced_rows is None and rec.readout is None


def test_readout_trace_names_a_preset_group_after_the_preset_and_lists_its_rows(tiny_atlas, store, sims):
    orn = tools.preset_selection(tiny_atlas, store, "orn", presets=PRESETS)["selection_id"]
    out = run(tiny_atlas, store, sims, readout_selection_id=orn)
    ro = sims.get(out["simulation_id"]).readout
    assert ro["label"] == "ORN" and ro["rows"] == [62]


def test_readout_rows_are_capped(tiny_atlas, store, sims):
    pen = ticket(tiny_atlas, store, cell_type="PEN_a")
    ro = sims.get(run(tiny_atlas, store, sims, readout_selection_id=pen)["simulation_id"]).readout
    assert ro["neurons"] == 60 and len(ro["rows"]) == 20


def test_a_baseline_is_never_evicted_while_a_silenced_run_still_points_at_it(tiny_atlas, store):
    svc = SimulationService.from_simulation(
        Simulation.from_arrays([62, 61, 60], [61, 60, 0], [1000, 1000, 1000], len(tiny_atlas.neurons)), max_results=3
    )
    epg = ticket(tiny_atlas, store, cell_type="EPG")
    pen = ticket(tiny_atlas, store, cell_type="PEN_a")
    first = run(tiny_atlas, store, svc, trials=1, silence_selection_id=epg, readout_selection_id=pen)["simulation_id"]
    for _ in range(4):
        run(tiny_atlas, store, svc, trials=1)
        for sim_id in list(svc._records):
            cid = svc._records[sim_id].compare_id
            assert cid is None or cid in svc._records, "dangling baseline"
    assert len(svc._records) <= 3
    # the pair goes together: once the silenced run is gone its baseline may go too
    with pytest.raises(UnknownSimulationError):
        svc.get(first)


# --- statistics (EB3) --------------------------------------------------------------------


def test_a_comparison_defaults_to_five_trials_and_a_plain_run_to_three(tiny_atlas, store, sims):
    orn = ticket(tiny_atlas, store, cell_type="ORN_DA1")
    epg = ticket(tiny_atlas, store, cell_type="EPG")
    plain = tools.stimulate(tiny_atlas, store, sims, orn, rate_hz=300, duration_ms=100)
    silenced = tools.stimulate(tiny_atlas, store, sims, orn, rate_hz=300, duration_ms=100, silence_selection_id=epg)
    assert plain["stimulus"]["trials"] == 3 and silenced["stimulus"]["trials"] == 5
    explicit = tools.stimulate(tiny_atlas, store, sims, orn, rate_hz=300, duration_ms=100, trials=2, silence_selection_id=epg)
    assert explicit["stimulus"]["trials"] == 2


def test_silencing_reports_the_readout_with_intervals_and_a_test(tiny_atlas, store, sims):
    epg = ticket(tiny_atlas, store, cell_type="EPG")
    pen = ticket(tiny_atlas, store, cell_type="PEN_a")
    out = run(tiny_atlas, store, sims, trials=5, silence_selection_id=epg, readout_selection_id=pen)
    st = out["readout_stats"]
    assert st["trials"] == 5
    assert st["baseline"]["ci_low"] <= st["baseline"]["mean_hz"] <= st["baseline"]["ci_high"]
    assert st["silenced"]["mean_hz"] == 0 and st["delta_hz"] == pytest.approx(-st["baseline"]["mean_hz"], abs=0.1)
    assert st["p_value"] is not None and st["p_value"] < 0.05
    assert "significant at p<0.05" in st["verdict"] and "model" in st["verdict"].lower()
    assert st["baseline"]["mean_hz"] == out["readout_change"]["baseline_mean_hz"]  # same numbers as the old summary
    assert len(json.dumps(out)) < 2600


def test_one_trial_says_it_cannot_test(tiny_atlas, store, sims):
    epg = ticket(tiny_atlas, store, cell_type="EPG")
    pen = ticket(tiny_atlas, store, cell_type="PEN_a")
    st = run(tiny_atlas, store, sims, trials=1, silence_selection_id=epg, readout_selection_id=pen)["readout_stats"]
    assert st["p_value"] is None and st["baseline"]["ci_low"] is None and "at least 2 trials" in st["verdict"]


def test_no_effect_is_called_noise_not_a_finding(tiny_atlas, store, sims):
    orn = ticket(tiny_atlas, store, cell_type="ORN_DA1")
    pen = ticket(tiny_atlas, store, cell_type="PEN_a")
    out = tools.stimulate(tiny_atlas, store, sims, orn, rate_hz=300, duration_ms=100, trials=3,
                          silence_selection_id=pen, readout_selection_id=ticket(tiny_atlas, store, cell_type="EPG"))
    st = out["readout_stats"]  # silencing PEN (a leaf here) cannot change EPG
    assert st["delta_hz"] == 0 and st["p_value"] == 1.0 and "not distinguishable from noise with 3 trials" in st["verdict"]


# --- runaway excitation warning ----------------------------------------------------------


def test_a_brain_wide_cascade_gets_a_runaway_warning(tiny_atlas, store, sims, monkeypatch):
    monkeypatch.setattr(tools, "RUNAWAY_ACTIVE", 3)  # the tiny net lights 3+ neurons; the real cut is 5000
    out = run(tiny_atlas, store, sims)
    warning = out["warning"]
    assert warning.startswith("likely runaway excitation")
    assert f"{out['active_neurons']} neurons" in warning
    assert "from 1 stimulated" in warning
    # the reason the model must give instead of calling it a biological amplifier
    assert "dopamine, octopamine and serotonin" in warning and "GABA and glutamate" in warning
    assert "artifact" in warning and "lower rate_hz" in warning


def test_ordinary_runs_carry_no_runaway_warning(tiny_atlas, store, sims):
    assert tools.RUNAWAY_ACTIVE == 5000
    assert "warning" not in run(tiny_atlas, store, sims)


def test_a_silenced_run_warns_when_either_side_ran_away(tiny_atlas, store, sims, monkeypatch):
    monkeypatch.setattr(tools, "RUNAWAY_ACTIVE", 3)
    stim = ticket(tiny_atlas, store, cell_type="ORN_DA1")
    silence = ticket(tiny_atlas, store, cell_type="PEN_a")  # silencing downstream keeps the upstream 3 lit
    out = tools.stimulate(tiny_atlas, store, sims, stim, rate_hz=300, duration_ms=200, trials=2,
                          silence_selection_id=silence)
    assert "warning" in out


def test_non_sensory_rows_drops_only_sensory_neurons(tiny_atlas):
    # rows 0..59 PEN_a (central), 60 EPG (central), 61 T4a (optic), 62 ORN_DA1 (sensory)
    assert tools.non_sensory_rows(tiny_atlas, np.array([0, 61, 62])).tolist() == [0, 61]
    assert tools.non_sensory_rows(tiny_atlas, np.array([], dtype=int)).tolist() == []
