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
                    "active_count": rec.active_count, "active_hz": 1.0, "compare_id": None,
                    "silenced_rows": [], "silenced_truncated": False, "readout": None}
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
    assert names == ["meta", "tool_call", "tool_result", "highlight", "tool_call", "tool_result", "highlight", "activity", "usage", "answer", "done"]
    act = dict(events)["activity"]
    rec = svc.get(act["simulation_id"])
    assert act == {"simulation_id": rec.id, "n_bins": rec.n_bins, "bin_ms": rec.bin_ms,
                   "active_count": rec.active_count, "active_hz": 1.0, "compare_id": None}
    assert rec.active_count is not None and rec.active_count <= rec.n_rows  # spiked at least once >= active > 1 Hz


# --- comparison runs (VL4) and rows metadata (VL5) -----------------------------------------


def compare_run(client, **extra):
    epg = client.get("/neurons/search", params={"cell_type": "EPG"}).json()["selection_id"]
    pen = client.get("/neurons/search", params={"cell_type": "PEN_a"}).json()["selection_id"]
    body = {"selection_id": orn_ticket(client), "silence_selection_id": epg, "readout_selection_id": pen, **BODY, **extra}
    return client.post("/simulate", json=body)


def test_simulate_with_silencing_exposes_both_recordings(tiny_tsv_path, svc):
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc)) as client:
        r = compare_run(client)
        assert r.status_code == 200
        meta = client.get(f"/simulations/{r.json()['simulation_id']}").json()
        base_meta = client.get(f"/simulations/{meta['compare_id']}").json()
        base_frames = client.get(f"/simulations/{meta['compare_id']}/frames")
    assert meta["compare_id"].startswith("sim_") and base_meta["compare_id"] is None
    assert meta["silenced_rows"] == [60] and base_meta["silenced_rows"] == []
    assert meta["silenced_truncated"] is False and base_meta["silenced_truncated"] is False
    assert meta["readout"]["label"] == "PEN_a" and len(meta["readout"]["hz"]) == meta["n_bins"]
    assert len(base_frames.content) == 4 * base_meta["n_rows"] + base_meta["n_rows"] * base_meta["n_bins"]


def test_too_many_silenced_rows_are_flagged_not_confused_with_no_silencing(tiny_tsv_path, svc):
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc, max_inline_indices=0)) as client:
        sim_id = compare_run(client).json()["simulation_id"]
        meta = client.get(f"/simulations/{sim_id}").json()
        plain = client.post("/simulate", json={"selection_id": orn_ticket(client), **BODY}).json()["simulation_id"]
        plain_meta = client.get(f"/simulations/{plain}").json()
    assert meta["silenced_truncated"] is True and meta["silenced_rows"] is None
    assert plain_meta["silenced_truncated"] is False and plain_meta["silenced_rows"] == []


def test_plain_meta_has_null_comparison_fields(tiny_tsv_path, svc):
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc)) as client:
        sim_id = client.post("/simulate", json={"selection_id": orn_ticket(client), **BODY}).json()["simulation_id"]
        meta = client.get(f"/simulations/{sim_id}").json()
    assert meta["compare_id"] is None and meta["silenced_rows"] == [] and meta["readout"] is None


def test_simulate_accepts_a_silence_preset_and_rejects_both(tiny_tsv_path, svc):
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc)) as client:
        both = client.post("/simulate", json={"selection_id": orn_ticket(client), "silence_preset": "sugar",
                                              "silence_selection_id": "sel_x", **BODY})
        unknown = client.post("/simulate", json={"selection_id": orn_ticket(client), "silence_preset": "nope", **BODY})
    assert both.status_code == 422 and "silence_preset" in both.json()["detail"]
    assert unknown.status_code == 422 and "Unknown preset" in unknown.json()["detail"]


def test_simulations_rows_lists_cell_type_and_super_class(tiny_tsv_path, svc):
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc)) as client:
        sim_id = client.post("/simulate", json={"selection_id": orn_ticket(client), **BODY}).json()["simulation_id"]
        everything = client.get(f"/simulations/{sim_id}/rows")
        subset = client.get(f"/simulations/{sim_id}/rows", params={"rows": "62,61,5000"})
        missing = client.get("/simulations/sim_nope/rows")
        bad = client.get(f"/simulations/{sim_id}/rows", params={"rows": "a,b"})
    rec = svc.get(sim_id)
    assert [r["row"] for r in everything.json()] == [int(x) for x in rec.rows]
    assert subset.json() == [{"row": 61, "cell_type": "T4a", "super_class": "optic"},
                             {"row": 62, "cell_type": "ORN_DA1", "super_class": "sensory"}]
    assert missing.status_code == 404 and bad.status_code == 422


def test_activity_event_carries_the_compare_id(tiny_tsv_path, svc):
    def silence_then_stimulate(msgs):
        last = msgs[-1]
        if last.type == "human":
            return call("search_neurons", {"cell_type": "ORN_DA1"})
        if last.type == "tool" and "simulation_id" not in text_of(last) and "ORN_DA1" in text_of(last):
            sid = json.loads(text_of(last))["selection_id"]
            return call("search_neurons", {"cell_type": "EPG"}, "c2")
        if last.type == "tool" and "simulation_id" not in text_of(last):
            epg = json.loads(text_of(last))["selection_id"]
            orn = next(json.loads(text_of(m))["selection_id"] for m in msgs if m.type == "tool")
            return call("stimulate", {"selection_id": orn, "silence_selection_id": epg, "rate_hz": 300,
                                      "duration_ms": 100, "trials": 1}, "c3")
        return AIMessage(content="done")

    client = TestClient(create_app(atlas_path=tiny_tsv_path, model=fn_model(silence_then_stimulate), simulation=svc))
    with client:
        events = parse_sse(client.post("/chat", json={"message": "silence EPG"}).text)
    act = dict(events)["activity"]
    assert act["compare_id"] == svc.get(act["simulation_id"]).compare_id and act["compare_id"]
