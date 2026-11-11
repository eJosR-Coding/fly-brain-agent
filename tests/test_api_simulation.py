"""Simulation endpoints and the activity SSE event, over a tiny network and a fake model."""

import json
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from flybrain.api import create_app
from flybrain.sim_service import SimulationService
from flybrain.simulation import Simulation
from fakes import call, fn_model
from test_api_chat import parse_sse

EDGES = [(62, 61, 1000), (61, 60, 1000), (60, 0, 1000)]
BODY = {"rate_hz": 300, "duration_ms": 100, "trials": 1}


@pytest.fixture
def svc(tiny_atlas):
    pre, post, w = (np.array(c) for c in zip(*EDGES))
    return SimulationService.from_simulation(Simulation.from_arrays(pre, post, w, n_neurons=len(tiny_atlas.neurons)))


def orn_ticket(client):
    return client.get("/neurons/search", params={"cell_type": "ORN_DA1"}).json()["selection_id"]


def test_health_reports_the_simulator(tiny_tsv_path, svc):
    with TestClient(create_app(atlas_path=tiny_tsv_path)) as off:
        assert off.get("/health").json()["simulation"] is False
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc)) as on:
        assert on.get("/health").json()["simulation"] is True


def test_simulate_returns_the_summary_and_a_simulation_id(tiny_tsv_path, svc):
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc)) as client:
        r = client.post("/simulate", json={"selection_id": orn_ticket(client), **BODY})
    assert r.status_code == 200
    body = r.json()
    assert body["simulation_id"].startswith("sim_") and body["active_neurons"] >= 3
    assert body["top_responders"][0]["cell_type"] and "not the living fly" in body["note"]


def test_simulate_with_a_readout_selection(tiny_tsv_path, svc):
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc)) as client:
        pen = client.get("/neurons/search", params={"cell_type": "PEN_a"}).json()["selection_id"]
        r = client.post("/simulate", json={"selection_id": orn_ticket(client), "readout_selection_id": pen, **BODY})
    assert r.json()["readout"]["neurons"] == 60


@pytest.mark.parametrize(
    "body",
    [{}, {"preset": "sugar", "selection_id": "sel_x"}, {"selection_id": "sel_x", "trials": "many"},
     {"selection_id": "sel_x", "trials": 2.5}, {"selection_id": "sel_x", "rate_hz": "fast"},
     {"selection_id": "sel_x", "bogus": 1}, {"preset": "nope"}, {"preset": "sugar"}],
)
def test_simulate_rejects_bad_input_with_422(tiny_tsv_path, svc, body):
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc)) as client:
        r = client.post("/simulate", json=body)
    assert r.status_code == 422, r.text


def test_unknown_preset_lists_the_available_ones(tiny_tsv_path, svc):
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc)) as client:
        detail = client.post("/simulate", json={"preset": "nope"}).json()["detail"]
    assert "sugar" in detail and "mn9" in detail


def test_unknown_selection_is_404(tiny_tsv_path, svc):
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc)) as client:
        r = client.post("/simulate", json={"selection_id": "sel_nope", **BODY})
    assert r.status_code == 404 and "unknown selection" in r.json()["detail"]


def test_simulate_without_weights_is_503_with_guidance(tiny_tsv_path):
    with TestClient(create_app(atlas_path=tiny_tsv_path)) as client:
        r = client.post("/simulate", json={"selection_id": orn_ticket(client), **BODY})
    assert r.status_code == 503 and "flybrain-build-sim-weights" in r.json()["detail"]


def test_simulate_does_not_block_the_event_loop(tiny_tsv_path, svc):
    started = threading.Event()
    real = svc.simulation().run

    def slow(*a, **kw):
        started.set()
        time.sleep(0.6)
        return real(*a, **kw)

    svc.simulation().run = slow
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc)) as client:
        sid = orn_ticket(client)
        worker = threading.Thread(target=lambda: client.post("/simulate", json={"selection_id": sid, **BODY}))
        worker.start()
        assert started.wait(2)
        t0 = time.perf_counter()
        assert client.get("/health").status_code == 200
        assert time.perf_counter() - t0 < 0.4  # answered while the simulation is still running
        worker.join()


def test_frames_endpoints(tiny_tsv_path, svc):
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc)) as client:
        summary = client.post("/simulate", json={"selection_id": orn_ticket(client), **BODY}).json()
        sim_id, active = summary["simulation_id"], summary["active_neurons"]
        meta = client.get(f"/simulations/{sim_id}").json()
        frames = client.get(f"/simulations/{sim_id}/frames")
    rec = svc.get(sim_id)
    assert meta == {"simulation_id": sim_id, "bin_ms": 1.0, "n_bins": 100, "n_rows": rec.n_rows, "duration_ms": 100.0,
                    "stimulus": {"neurons": 1, "rate_hz": 300.0, "duration_ms": 100.0, "trials": 1}, "trials": 1,
                    "active_count": rec.active_count, "active_hz": 1.0}
    assert meta["active_count"] == active  # same "> 1 Hz" count the stimulate summary reports
    assert frames.headers["content-type"] == "application/octet-stream"
    raw = frames.content
    assert len(raw) == 4 * meta["n_rows"] + meta["n_rows"] * meta["n_bins"]
    rows = np.frombuffer(raw[: 4 * meta["n_rows"]], dtype="<u4")
    counts = np.frombuffer(raw[4 * meta["n_rows"]:], dtype=np.uint8).reshape(meta["n_rows"], meta["n_bins"])
    assert 62 in rows and (rows == rec.rows).all() and (counts == rec.frames).all() and counts.sum() > 0


def test_unknown_or_expired_simulation_is_404_with_a_message(tiny_tsv_path, svc):
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc)) as client:
        for path in ("/simulations/sim_nope", "/simulations/sim_nope/frames"):
            r = client.get(path)
            assert r.status_code == 404 and "unknown simulation" in r.json()["detail"]


def text_of(message) -> str:
    c = message.content
    return c if isinstance(c, str) else "".join(b["text"] if isinstance(b, dict) else b for b in c)


def search_then_stimulate(msgs):
    last = msgs[-1]
    if last.type == "human":
        return call("search_neurons", {"cell_type": "ORN_DA1"})
    if last.type == "tool" and "simulation_id" not in text_of(last):
        sid = json.loads(text_of(last))["selection_id"]
        return call("stimulate", {"selection_id": sid, "rate_hz": 300, "duration_ms": 100, "trials": 1}, "c2")
    return AIMessage(content="The model says three neurons respond.")


def test_chat_emits_an_activity_event_after_the_highlight(tiny_tsv_path, svc):
    client = TestClient(create_app(atlas_path=tiny_tsv_path, model=fn_model(search_then_stimulate), simulation=svc))
    with client:
        events = parse_sse(client.post("/chat", json={"message": "give it a smell"}).text)
    names = [n for n, _ in events]
    assert names == ["meta", "tool_call", "tool_result", "highlight", "tool_call", "tool_result", "highlight", "activity", "answer", "done"]
    act = dict(events)["activity"]
    rec = svc.get(act["simulation_id"])
    assert act == {"simulation_id": rec.id, "n_bins": rec.n_bins, "bin_ms": rec.bin_ms,
                   "active_count": rec.active_count, "active_hz": 1.0}
    assert rec.active_count is not None and rec.active_count <= rec.n_rows  # spiked at least once >= active > 1 Hz
