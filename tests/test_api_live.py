"""/ws/live round trips with TestClient and fake brain/body (no flygym, no GPU)."""

import sys
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from flybrain import body_runner
from flybrain.api import create_app
from flybrain.live import LiveSession, frames_from_bytes
from flybrain.sim_service import SimulationService, SimulationUnavailable
from flybrain.simulation import ChunkSpikes, Simulation, SimulationError
from test_live_session import GROUPS, Clock, FakeBody, FakeBrain


class Probe:
    """What the fake factory built, to assert on after the socket closed."""

    def __init__(self):
        self.sessions = []
        self.requests = []


def fake_factory(probe, error=None):
    def factory(atlas, row_index, sims, body_lock, **kw):
        probe.requests.append(kw)
        if error is not None:
            raise error
        clock = Clock()
        brain = FakeBrain(clock, script={i: {0: 2, 1: 2} for i in range(10_000)}, cost_s=0.0)
        body = FakeBody(clock, cost_s=0.0)
        orig = body.advance

        def slow(ms, drive, escape):
            time.sleep(0.003)
            return orig(ms, drive, escape)

        body.advance = slow
        s = LiveSession(brain, body, GROUPS, clock=clock, binary_frames=kw.get("binary_frames", False))
        probe.sessions.append(s)
        return s

    return factory


@pytest.fixture
def svc(tiny_atlas):
    pre, post, w = (np.array(c) for c in zip(*[(0, 1, 1000)]))
    return SimulationService.from_simulation(Simulation.from_arrays(pre, post, w, n_neurons=len(tiny_atlas.neurons)))


def app_with(tiny_tsv_path, svc, probe, error=None, **kw):
    return create_app(atlas_path=tiny_tsv_path, simulation=svc, live_factory=fake_factory(probe, error), **kw)


def wait_for(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def recv_until(ws, kind, limit=200):
    for _ in range(limit):
        m = ws.receive_json()
        if m["type"] == kind:
            return m
    raise AssertionError(f"no {kind!r} message")


def test_round_trip_start_chunks_stimulus_stop(tiny_tsv_path, svc):
    probe = Probe()
    with TestClient(app_with(tiny_tsv_path, svc, probe)) as c, c.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "start", "preset": None, "rate_hz": 0})
        started = ws.receive_json()
        assert started["type"] == "started" and started["chunk_ms"] == 50.0 and "header" in started
        first = recv_until(ws, "chunk")
        assert first["seq"] == 0 and first["backend"] == "cpu" and "speed_label" in first
        ws.send_json({"type": "stimulus", "rows": [3, 4], "rate_hz": 150})
        ack = recv_until(ws, "stimulus")
        assert ack["ok"] is True and ack["rows"] == 2 and ack["rate_hz"] == 150
        assert probe.sessions[0].brain.stimuli[-1] == ([3, 4], 150.0)
        ws.send_json({"type": "stimulus", "preset": None, "rate_hz": 0})
        assert recv_until(ws, "stimulus")["rows"] == 0
        ws.send_json({"type": "stop"})
        stopped = recv_until(ws, "stopped")
        assert stopped["reason"] == "client"
    assert wait_for(lambda: probe.sessions[0].brain.closed)
    assert not c.app.state.live_lock.locked()


def test_start_options_reach_the_factory_and_initial_stimulus_is_applied(tiny_tsv_path, svc):
    probe = Probe()
    with TestClient(app_with(tiny_tsv_path, svc, probe)) as c, c.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "start", "rows": [1, 2], "rate_hz": 100, "seed": 5, "include_active": True})
        assert ws.receive_json()["type"] == "started"
        assert probe.requests[0]["seed"] == 5 and probe.requests[0]["include_active"] is True
        rows, rate = probe.sessions[0].brain.stimuli[-1]
        assert rate == 100.0 and rows is not None
        ws.send_json({"type": "stop"})


def test_unknown_preset_and_bad_start_close_with_4400(tiny_tsv_path, svc):
    probe = Probe()
    with TestClient(app_with(tiny_tsv_path, svc, probe)) as c:
        for first in ({"type": "hello"}, {"type": "start", "preset": "nope", "rate_hz": 10},
                      {"type": "start", "preset": None, "rate_hz": -5}, {"type": "start", "rate_hz": "fast"}):
            with c.websocket_connect("/ws/live") as ws:
                ws.send_json(first)
                err = ws.receive_json()
                assert err["type"] == "error"
                with pytest.raises(WebSocketDisconnect) as e:
                    ws.receive_json()
                assert e.value.code == 4400
        assert not probe.sessions


def test_bad_stimulus_is_an_error_message_and_the_session_keeps_streaming(tiny_tsv_path, svc):
    probe = Probe()
    with TestClient(app_with(tiny_tsv_path, svc, probe)) as c, c.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "start", "preset": None, "rate_hz": 0})
        recv_until(ws, "chunk")
        for bad in ({"type": "stimulus", "rows": [9999], "rate_hz": 10}, {"type": "stimulus", "preset": "nope", "rate_hz": 1},
                    {"type": "stimulus", "rows": [1], "rate_hz": "x"}, {"type": "mystery"}):
            ws.send_json(bad)
            assert recv_until(ws, "error")["detail"]
        recv_until(ws, "chunk")
        ws.send_json({"type": "stop"})


def test_only_one_live_session_at_a_time(tiny_tsv_path, svc):
    probe = Probe()
    with TestClient(app_with(tiny_tsv_path, svc, probe)) as c, c.websocket_connect("/ws/live") as ws1:
        ws1.send_json({"type": "start", "preset": None, "rate_hz": 0})
        recv_until(ws1, "chunk")
        with c.websocket_connect("/ws/live") as ws2:
            with pytest.raises(WebSocketDisconnect) as e:
                ws2.receive_json()
            assert e.value.code == 4409 and "already" in e.value.reason
        recv_until(ws1, "chunk")  # the first one is unharmed
        ws1.send_json({"type": "stop"})
        recv_until(ws1, "stopped")
    with TestClient(app_with(tiny_tsv_path, svc, Probe())) as c2, c2.websocket_connect("/ws/live") as ws3:
        ws3.send_json({"type": "start", "preset": None, "rate_hz": 0})
        assert ws3.receive_json()["type"] == "started"


def test_disconnect_stops_the_session_and_frees_the_lock(tiny_tsv_path, svc):
    probe = Probe()
    with TestClient(app_with(tiny_tsv_path, svc, probe)) as c:
        with c.websocket_connect("/ws/live") as ws:
            ws.send_json({"type": "start", "preset": None, "rate_hz": 0})
            recv_until(ws, "chunk")
        assert wait_for(lambda: probe.sessions[0].brain.closed and not c.app.state.live_lock.locked())
        n = probe.sessions[0].seq
        time.sleep(0.1)
        assert probe.sessions[0].seq == n  # no chunks after the client left


def test_idle_session_is_stopped_after_the_idle_limit(tiny_tsv_path, svc):
    probe = Probe()
    with TestClient(app_with(tiny_tsv_path, svc, probe, live_idle_s=0.3)) as c, c.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "start", "preset": None, "rate_hz": 0})
        stopped = recv_until(ws, "stopped", limit=5000)
        assert stopped["reason"] == "idle"
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
        assert e.value.code == 4408
    assert wait_for(lambda: probe.sessions[0].brain.closed)


def test_pings_keep_a_session_alive(tiny_tsv_path, svc):
    probe = Probe()
    with TestClient(app_with(tiny_tsv_path, svc, probe, live_idle_s=0.4)) as c, c.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "start", "preset": None, "rate_hz": 0})
        end = time.time() + 0.9
        while time.time() < end:
            ws.send_json({"type": "ping"})
            ws.receive_json()
            time.sleep(0.05)
        ws.send_json({"type": "stop"})
        assert recv_until(ws, "stopped")["reason"] == "client"


def test_missing_body_or_weights_close_with_4503_and_the_reason(tiny_tsv_path, svc):
    for error, needle in ((body_runner.BodyUnavailable("needs flygym; run `uv sync --extra body`"), "uv sync --extra body"),
                          (SimulationUnavailable("weights missing"), "weights")):
        with TestClient(app_with(tiny_tsv_path, svc, Probe(), error=error)) as c, c.websocket_connect("/ws/live") as ws:
            ws.send_json({"type": "start", "preset": None, "rate_hz": 0})
            err = ws.receive_json()
            assert err["type"] == "error" and needle in err["detail"]
            with pytest.raises(WebSocketDisconnect) as e:
                ws.receive_json()
            assert e.value.code == 4503 and needle in e.value.reason
            assert not c.app.state.live_lock.locked()


def test_default_factory_reports_missing_flygym_as_4503(tiny_tsv_path, svc, monkeypatch):
    monkeypatch.setitem(sys.modules, "flygym", None)
    with TestClient(create_app(atlas_path=tiny_tsv_path, simulation=svc)) as c, c.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "start", "preset": None, "rate_hz": 0})
        assert "uv sync --extra body" in ws.receive_json()["detail"]
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
        assert e.value.code == 4503


def test_binary_frames_arrive_as_a_second_message(tiny_tsv_path, svc):
    probe = Probe()
    with TestClient(app_with(tiny_tsv_path, svc, probe)) as c, c.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "start", "preset": None, "rate_hz": 0, "binary_frames": True})
        assert ws.receive_json()["binary_frames"] is True
        msg = ws.receive_json()
        blob = ws.receive_bytes()
        assert msg["type"] == "chunk" and msg["frames"] is None
        assert frames_from_bytes(blob, msg["frames_binary"]["shape"]).shape == (10, 48, 7)
        ws.send_json({"type": "stop"})


def test_a_failing_session_reports_an_error_and_closes_1011(tiny_tsv_path, svc):
    probe = Probe()
    app = app_with(tiny_tsv_path, svc, probe)
    with TestClient(app) as c, c.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "start", "preset": None, "rate_hz": 0})
        recv_until(ws, "chunk")
        probe.sessions[0].brain.fail = RuntimeError("boom")
        assert "boom" in recv_until(ws, "error")["detail"]
        with pytest.raises(WebSocketDisconnect) as e:
            while True:
                ws.receive_json()
        assert e.value.code == 1011
    assert wait_for(lambda: not app.state.live_lock.locked())


def test_cross_origin_browsers_are_refused(tiny_tsv_path, svc):
    with TestClient(app_with(tiny_tsv_path, svc, Probe())) as c:
        with c.websocket_connect("/ws/live", headers={"origin": "http://evil.example"}) as ws:
            with pytest.raises(WebSocketDisconnect) as e:
                ws.receive_json()
            assert e.value.code == 4403
        with c.websocket_connect("/ws/live", headers={"origin": "http://localhost:5173"}) as ws:
            ws.send_json({"type": "start", "preset": None, "rate_hz": 0})
            assert ws.receive_json()["type"] == "started"
            ws.send_json({"type": "stop"})


def test_health_gpu_reports_thermal_and_live(tiny_tsv_path, svc):
    with TestClient(app_with(tiny_tsv_path, svc, Probe())) as c:
        h = c.get("/health/gpu").json()
        assert h["thermal"] is None and h["live"] == {"active": False, "stream_note": None}
        svc.thermal = __import__("flybrain.gpu_simulation", fromlist=["ThermalGuard"]).ThermalGuard(
            read=lambda: {"temperature_c": 83.0}, interval_s=0.0)
        svc.thermal.allow()
        h = c.get("/health/gpu").json()
        assert h["thermal"]["hot"] is True and h["thermal"]["limit_c"] == 80.0
