"""Experiment endpoints, chart and progress events, over a tiny network and a fake model."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from flybrain.api import create_app
from flybrain.sim_service import SimulationService
from flybrain.simulation import Simulation
from fakes import call, fn_model
from test_api_chat import parse_sse

EDGES = [(62, 61, 1000), (61, 60, 1000), (62, 5, 1000), (60, 0, 1000)]


@pytest.fixture
def svc(tiny_atlas):
    pre, post, w = (np.array(c) for c in zip(*EDGES))
    return SimulationService.from_simulation(Simulation.from_arrays(pre, post, w, n_neurons=len(tiny_atlas.neurons)))


def ticket(client, **q):
    return client.get("/neurons/search", params=q).json()["selection_id"]


DOSE = {"rates_hz": [10, 100, 300], "duration_ms": 100, "trials": 2}
SCREEN = {"rate_hz": 300, "duration_ms": 100, "trials": 3, "n": 3}


def test_dose_endpoint_runs_without_an_llm_and_inlines_the_chart(tiny_tsv_path, svc):
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc)) as client:
        r = client.post("/experiments/dose_response", json={
            "selection_id": ticket(client, cell_type="ORN_DA1"), "readout_selection_id": ticket(client, cell_type="EPG"), **DOSE})
    assert r.status_code == 200
    body = r.json()
    assert [p["rate_hz"] for p in body["points"]] == [10, 100, 300]
    assert body["chart"]["kind"] == "dose_response" and body["chart"]["id"] == body["chart_id"]
    assert body["chart"]["readout_label"] == "EPG"


def test_screen_endpoint_returns_rows_and_the_chart(tiny_tsv_path, svc):
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc)) as client:
        r = client.post("/experiments/screen", json={
            "selection_id": ticket(client, cell_type="ORN_DA1"), "readout_selection_id": ticket(client, cell_type="EPG"), **SCREEN})
        body = r.json()
        first = body["chart"]["rows"][0]
        meta = client.get(f"/simulations/{first['simulation_id']}").json()  # a row opens its diff map
    assert r.status_code == 200 and body["chart"]["kind"] == "screen" and body["significant_count"] >= 1
    assert meta["compare_id"] == body["baseline_simulation_id"] and meta["silenced_rows"] == [first["row"]]


def test_experiment_endpoints_accept_presets_and_reject_bad_input(tiny_tsv_path, svc):
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc)) as client:
        orn, epg = ticket(client, cell_type="ORN_DA1"), ticket(client, cell_type="EPG")
        both = client.post("/experiments/dose_response", json={"preset": "sugar", "selection_id": orn, "readout_selection_id": epg})
        none = client.post("/experiments/dose_response", json={"readout_selection_id": epg})
        no_readout = client.post("/experiments/screen", json={"selection_id": orn})
        unknown = client.post("/experiments/screen", json={"selection_id": orn, "readout_preset": "nope"})
        gone = client.post("/experiments/dose_response", json={"selection_id": "sel_nope", "readout_selection_id": epg})
        bad = client.post("/experiments/dose_response", json={"selection_id": orn, "readout_selection_id": epg, "rates_hz": "fast"})
        extra = client.post("/experiments/screen", json={"selection_id": orn, "readout_selection_id": epg, "bogus": 1})
        mode = client.post("/experiments/screen", json={"selection_id": orn, "readout_selection_id": epg, "candidates": "all"})
    assert both.status_code == 422 and "not both" in both.json()["detail"]
    assert none.status_code == 422 and no_readout.status_code == 422
    assert unknown.status_code == 422 and "Unknown preset" in unknown.json()["detail"]
    assert gone.status_code == 404 and bad.status_code == 422 and extra.status_code == 422
    assert mode.status_code == 422 and "top_non_sensory" in mode.json()["detail"]


def test_experiments_without_weights_are_503(tiny_tsv_path):
    with TestClient(create_app(atlas_path=tiny_tsv_path)) as client:
        r = client.post("/experiments/screen", json={"selection_id": "sel_x", "readout_selection_id": "sel_y"})
    assert r.status_code == 503 and "flybrain-build-sim-weights" in r.json()["detail"]


def test_progress_endpoint_reports_the_running_experiment_or_idle(tiny_tsv_path, svc):
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc)) as client:
        assert client.get("/experiments/progress").json() == {"running": False}
        svc.report_progress("silencing_screen", 3, 8)
        assert client.get("/experiments/progress").json() == {
            "running": True, "experiment": "silencing_screen", "done": 3, "total": 8}
        svc.finish_progress()
        assert client.get("/experiments/progress").json() == {"running": False}


def tool_texts(msgs):
    return [m.content if isinstance(m.content, str) else "".join(b.get("text", "") for b in m.content) for m in msgs if m.type == "tool"]


def run_experiment_turn(tool, extra):
    def reply(msgs):
        done = tool_texts(msgs)
        if not done:
            return call("search_neurons", {"cell_type": "ORN_DA1"}, "c1")
        if len(done) == 1:
            return call("search_neurons", {"cell_type": "EPG"}, "c2")
        if len(done) == 2:
            orn, epg = (json.loads(t)["selection_id"] for t in done)
            return call(tool, {"selection_id": orn, "readout_selection_id": epg, **extra}, "c3")
        return AIMessage(content="The model predicts a graded response.")

    return reply


def test_chat_streams_progress_then_a_chart_event_for_a_dose_response(tiny_tsv_path, svc):
    reply = run_experiment_turn("dose_response", DOSE)
    with TestClient(create_app(atlas_path=tiny_tsv_path, model=fn_model(reply), simulation=svc)) as client:
        events = parse_sse(client.post("/chat", json={"message": "dose"}).text)
    names = [n for n, _ in events]
    last_call = max(i for i, n in enumerate(names) if n == "tool_call")
    tail = names[last_call:]
    assert tail[0] == "tool_call" and tail.count("progress") == 4 and tail.index("progress") < tail.index("tool_result")
    assert tail.index("tool_result") < tail.index("chart") < tail.index("answer")
    progress = [d for n, d in events if n == "progress"]
    assert [(p["experiment"], p["done"], p["total"]) for p in progress] == [("dose_response", i, 3) for i in range(4)]
    chart = dict(events)["chart"]
    assert chart["kind"] == "dose_response" and len(chart["points"]) == 3 and chart["id"].startswith("chart_")


def test_chat_emits_the_screen_chart_with_rows_that_open_diff_maps(tiny_tsv_path, svc):
    reply = run_experiment_turn("silencing_screen", SCREEN)
    with TestClient(create_app(atlas_path=tiny_tsv_path, model=fn_model(reply), simulation=svc)) as client:
        events = parse_sse(client.post("/chat", json={"message": "screen"}).text)
    chart = dict(events)["chart"]
    assert chart["kind"] == "screen" and chart["rows"][0]["label"] == "T4a" and chart["rows"][0]["significant"] is True
    assert svc.get(chart["rows"][0]["simulation_id"]).compare_id == chart["baseline_simulation_id"]
    assert sum(1 for n, _ in events if n == "progress") == 4  # 0..3 of 3 candidates


def test_a_chat_turn_with_no_experiment_has_no_progress_or_chart(tiny_tsv_path, svc):
    def reply(msgs):
        return call("search_neurons", {"cell_type": "EPG"}) if msgs[-1].type == "human" else AIMessage(content="ok")

    with TestClient(create_app(atlas_path=tiny_tsv_path, model=fn_model(reply), simulation=svc)) as client:
        names = [n for n, _ in parse_sse(client.post("/chat", json={"message": "x"}).text)]
    assert "progress" not in names and "chart" not in names


def test_experiment_tools_get_extra_turn_time_from_the_configured_budget(tiny_tsv_path, svc, monkeypatch):
    monkeypatch.setenv("FLYBRAIN_EXPERIMENT_TIMEOUT", "123")
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc)) as client:
        assert client.app.state.experiment_timeout == 123.0
    monkeypatch.setenv("FLYBRAIN_EXPERIMENT_TIMEOUT", "-1")
    with pytest.raises(ValueError, match="FLYBRAIN_EXPERIMENT_TIMEOUT"):
        create_app(atlas_path=tiny_tsv_path)
    monkeypatch.delenv("FLYBRAIN_EXPERIMENT_TIMEOUT")
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc, experiment_timeout=7)) as client:
        assert client.app.state.experiment_timeout == 7.0
