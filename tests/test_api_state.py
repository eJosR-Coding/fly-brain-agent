"""API-side state machines: one running turn per conversation (S12), the experiment progress
window (S14) and the chat flag following the tool server (S15). Fake model, no network."""

import json
import threading

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from flybrain.api import create_app
from fakes import call, fn_model
from test_api_chat import chat, parse_sse
from test_api_experiments import DOSE, EDGES, svc, tool_texts  # noqa: F401  (svc is a fixture)

BUSY = "a turn is already running on this conversation"


class Gate:
    """A model that blocks on messages containing 'slow' until released."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def reply(self, msgs):
        if "slow" in msgs[-1].text:
            self.entered.set()
            assert self.release.wait(10), "gate never released"
        return AIMessage(content=f"answer to {msgs[-1].text}")


def _in_thread(client, **kw):
    out = {}

    def run():
        out["r"] = chat(client, **kw)

    t = threading.Thread(target=run)
    t.start()
    return t, out


# --- S12: one turn at a time per conversation ----------------------------------------------

def test_sequential_turns_on_one_conversation_both_run(tiny_tsv_path):
    app = create_app(atlas_path=tiny_tsv_path, model=fn_model(lambda m: AIMessage(content="ok")))
    with TestClient(app) as client:
        first = parse_sse(chat(client, "one", conversation_id="c").text)
        second = parse_sse(chat(client, "two", conversation_id="c").text)
        assert app.state.turns == set()
    for events in (first, second):
        assert [n for n, _ in events] == ["meta", "usage", "answer", "done"]


def test_a_second_concurrent_turn_is_rejected_and_the_first_is_unaffected(tiny_tsv_path):
    gate = Gate()
    app = create_app(atlas_path=tiny_tsv_path, model=fn_model(gate.reply))
    with TestClient(app) as client:
        t, out = _in_thread(client, message="slow one", conversation_id="c")
        assert gate.entered.wait(10)
        busy = parse_sse(chat(client, "second", conversation_id="c").text)
        gate.release.set()
        t.join(10)
        first = parse_sse(out["r"].text)
        after = parse_sse(chat(client, "third", conversation_id="c").text)  # the owner left: free again
    assert [n for n, _ in busy] == ["meta", "error", "done"]
    assert busy[0][1]["conversation_id"] == "c" and busy[1][1] == {"message": BUSY}
    assert dict(first)["answer"] == {"text": "answer to slow one"}
    assert dict(after)["answer"] == {"text": "answer to third"}


def test_different_conversations_run_concurrently(tiny_tsv_path):
    gate = Gate()
    app = create_app(atlas_path=tiny_tsv_path, model=fn_model(gate.reply))
    with TestClient(app) as client:
        t, out = _in_thread(client, message="slow one", conversation_id="a")
        assert gate.entered.wait(10)
        other = parse_sse(chat(client, "quick", conversation_id="b").text)
        gate.release.set()
        t.join(10)
    assert dict(other)["answer"] == {"text": "answer to quick"}


def test_the_conversation_is_released_after_a_timeout(tiny_tsv_path):
    import time

    def slow_then_fast(msgs):
        if "slow" in msgs[-1].text:
            time.sleep(1)
        return AIMessage(content="ok")

    app = create_app(atlas_path=tiny_tsv_path, model=fn_model(slow_then_fast), turn_timeout=0.3)
    with TestClient(app) as client:
        timed_out = parse_sse(chat(client, "slow", conversation_id="c").text)
        assert app.state.turns == set()
        again = parse_sse(chat(client, "fast", conversation_id="c").text)
    assert "took longer" in dict(timed_out)["error"]["message"]
    assert "error" not in dict(again) and dict(again)["answer"] == {"text": "ok"}


def test_the_conversation_is_released_after_a_model_error(tiny_tsv_path):
    def boom(msgs):
        raise RuntimeError("provider exploded")

    app = create_app(atlas_path=tiny_tsv_path, model=fn_model(boom))
    with TestClient(app) as client:
        first = parse_sse(chat(client, "x", conversation_id="c").text)
        second = parse_sse(chat(client, "y", conversation_id="c").text)
    for events in (first, second):
        assert "exploded" in dict(events)["error"]["message"]  # never the busy message


def test_a_rejected_turn_does_not_release_the_owners_claim(tiny_tsv_path):
    gate = Gate()
    app = create_app(atlas_path=tiny_tsv_path, model=fn_model(gate.reply))
    with TestClient(app) as client:
        t, _ = _in_thread(client, message="slow one", conversation_id="c")
        assert gate.entered.wait(10)
        chat(client, "second", conversation_id="c")
        assert app.state.turns == {"c"}
        gate.release.set()
        t.join(10)
        assert app.state.turns == set()


# --- S14: progress follows the experiments actually in flight ------------------------------

def test_progress_is_not_dropped_by_a_parallel_quick_tool(tiny_tsv_path, svc):  # noqa: F811
    def reply(msgs):
        done = tool_texts(msgs)
        if not done:
            return call("search_neurons", {"cell_type": "ORN_DA1"}, "c1")
        if len(done) == 1:
            return call("search_neurons", {"cell_type": "EPG"}, "c2")
        if len(done) == 2:
            orn, epg = (json.loads(t)["selection_id"] for t in done)
            args = {"selection_id": orn, "readout_selection_id": epg, **DOSE}
            return AIMessage(content="", tool_calls=[
                {"name": "dose_response", "args": args, "id": "c3"},
                {"name": "search_neurons", "args": {"cell_type": "EPG"}, "id": "c4"},
            ])
        return AIMessage(content="done")

    with TestClient(create_app(atlas_path=tiny_tsv_path, model=fn_model(reply), simulation=svc)) as client:
        events = parse_sse(chat(client, "dose and search").text)
    assert sum(1 for n, _ in events if n == "progress") == 4
    assert "chart" in dict(events)


# --- S15: chat follows the tool server -----------------------------------------------------

def _fake_stream(event):
    async def stream(*a, **kw):
        yield event

    return stream


def test_health_flips_chat_off_once_the_tool_server_is_reported_dead(tiny_tsv_path, monkeypatch):
    import flybrain.api as api

    app = create_app(atlas_path=tiny_tsv_path, model=fn_model(lambda m: AIMessage(content="ok")), model_name="m:1")
    dead = {"type": "error", "message": "the flybrain server stopped: Connection closed", "tools_stopped": True}
    with TestClient(app) as client:
        assert client.get("/health").json()["chat"] is True
        monkeypatch.setattr(api, "stream_events", _fake_stream(dead))
        events = parse_sse(chat(client).text)
        body = client.get("/health").json()
    assert dict(events)["error"] == {"message": dead["message"]}  # SSE shape unchanged
    assert body["chat"] is False and body["model"] is None and set(body) >= {"status", "neurons", "simulation"}


def test_other_errors_with_the_same_words_do_not_flip_chat(tiny_tsv_path, monkeypatch):
    import flybrain.api as api

    app = create_app(atlas_path=tiny_tsv_path, model=fn_model(lambda m: AIMessage(content="ok")), model_name="m:1")
    lookalike = {"type": "error", "message": "the flybrain server stopped: but only said by a model"}
    with TestClient(app) as client:
        monkeypatch.setattr(api, "stream_events", _fake_stream(lookalike))
        chat(client)
        assert client.get("/health").json()["chat"] is True
