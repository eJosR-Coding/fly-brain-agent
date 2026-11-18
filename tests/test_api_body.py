"""POST /body/from-simulation/{id}: recording -> DN drive -> physics clip (the runner is monkeypatched here)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from flybrain import body_runner, body_service
from flybrain.api import create_app
from flybrain.sim_service import SimulationService
from flybrain.simulation import Simulation


@pytest.fixture
def svc(tiny_atlas):
    pre, post, w = (np.array(c) for c in zip(*[(0, 1, 1000)]))
    return SimulationService.from_simulation(Simulation.from_arrays(pre, post, w, n_neurons=len(tiny_atlas.neurons)))


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake(drive, bin_ms, duration_ms, seed=0, escape=None):
        seen.append({"drive": drive, "bin_ms": bin_ms, "duration_ms": duration_ms, "seed": seed})
        return {"format": "fly-walk/1", "n_frames": 3}

    monkeypatch.setattr(body_runner, "run_body", fake)
    return seen


def client_with_record(tiny_tsv_path, svc, rows=(), frames=None, bin_ms=100.0, duration_ms=1000.0):
    c = TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc))
    c.__enter__()
    frames = np.zeros((len(rows), 10), np.uint8) if frames is None else np.asarray(frames, np.uint8)
    sid = svc.add(np.asarray(rows, np.uint32), frames, bin_ms, duration_ms, 1, {"kind": "test"})
    return c, sid


def test_happy_path_returns_clip_drive_caption_and_note(tiny_tsv_path, svc, calls):
    c, sid = client_with_record(tiny_tsv_path, svc)
    with c:
        r = c.post(f"/body/from-simulation/{sid}", json={"duration_ms": 500, "seed": 7})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"clip", "drive", "caption", "note"} and body["clip"]["format"] == "fly-walk/1"
    assert body["drive"]["duration_ms"] == 500 and "bin_ms" in body["drive"] and "our" in body["caption"].lower()
    assert "OUR" in body["note"] and "8 s" in body["note"]
    assert calls[0]["duration_ms"] == 500 and calls[0]["seed"] == 7 and calls[0]["bin_ms"] == 100.0


def test_body_is_optional_and_defaults_to_the_recording_capped_at_3_seconds(tiny_tsv_path, svc, calls):
    c, sid = client_with_record(tiny_tsv_path, svc, duration_ms=1000.0)
    with c:
        assert c.post(f"/body/from-simulation/{sid}").status_code == 200
        assert calls[-1]["duration_ms"] == 1000.0 and calls[-1]["seed"] == 0
        assert c.post(f"/body/from-simulation/{sid}", json={"duration_ms": 2000}).status_code == 200
        assert calls[-1]["duration_ms"] == 1000.0  # never beyond the recording
    c2, sid2 = client_with_record(tiny_tsv_path, svc, duration_ms=9000.0)
    with c2:
        c2.post(f"/body/from-simulation/{sid2}")
    assert calls[-1]["duration_ms"] == body_service.MAX_BODY_MS


def test_unknown_simulation_is_404_and_bad_input_is_422(tiny_tsv_path, svc, calls):
    c, sid = client_with_record(tiny_tsv_path, svc)
    with c:
        assert c.post("/body/from-simulation/nope").status_code == 404
        for bad in ({"duration_ms": 0}, {"duration_ms": -5}, {"duration_ms": 3001}, {"seed": 1.5}, {"x": 1},
                    {"duration_ms": "long"}):
            assert c.post(f"/body/from-simulation/{sid}", json=bad).status_code == 422, bad
    assert calls == []


def test_missing_extra_is_503_with_the_install_hint(tiny_tsv_path, svc, monkeypatch):
    def unavailable(*a, **k):
        raise body_runner.BodyUnavailable(f"install it with `{body_runner.INSTALL_HINT}`")

    monkeypatch.setattr(body_runner, "run_body", unavailable)
    c, sid = client_with_record(tiny_tsv_path, svc)
    with c:
        r = c.post(f"/body/from-simulation/{sid}")
    assert r.status_code == 503 and "uv sync --extra body" in r.json()["detail"]


