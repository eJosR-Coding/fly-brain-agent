"""POST /chat over SSE with the real in-memory MCP server and a fake model. No network."""

import json

import numpy as np
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from flybrain.api import create_app
from fakes import call, fn_model


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.replace("\r\n", "\n").strip().split("\n\n"):
        name, data = None, None
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        if name:
            events.append((name, data))
    return events


def search_then_answer(msgs):
    if msgs[-1].type == "human":
        return call("search_neurons", {"nt": "glutamate"})
    return AIMessage(content="One glutamate neuron.")


def chat(client, message="hi", **extra):
    return client.post("/chat", json={"message": message, **extra})


def make_client(path, reply, **kw):
    model = fn_model(reply)
    return TestClient(create_app(atlas_path=path, model=model, **kw)), model


def test_event_order_and_highlight_indices(tiny_tsv_path, tiny_atlas):
    client, _ = make_client(tiny_tsv_path, search_then_answer)
    with client:
        r = chat(client, "glutamate?")
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert [n for n, _ in events] == ["meta", "tool_call", "highlight", "answer", "done"]
    assert events[0][1]["conversation_id"]
    assert events[1][1] == {"tool": "search_neurons", "args": {"nt": "glutamate"}}
    hl = events[2][1]
    expected = tiny_atlas.neurons.index[tiny_atlas.neurons["top_nt"] == "glutamate"].tolist()
    assert hl["indices"] == expected == [60]
    assert hl["total"] == 1 and hl["selection_id"].startswith("sel_")
    assert events[3][1] == {"text": "One glutamate neuron."}


def test_big_selection_sends_fetch_url_instead_of_indices(tiny_tsv_path):
    client, _ = make_client(tiny_tsv_path, lambda m: call("search_neurons", {}) if m[-1].type == "human" else AIMessage(content="ok"), max_inline_indices=10)
    with client:
        events = parse_sse(chat(client).text)
        hl = dict(events)["highlight"]
        assert hl["indices"] is None and hl["total"] == 63
        assert hl["fetch"] == f"/selections/{hl['selection_id']}/indices"
        body = client.get(hl["fetch"])
        assert body.headers["content-type"].startswith("application/octet-stream")
        assert np.frombuffer(body.content, "<u4").tolist() == list(range(63))


def test_second_turn_sees_previous_messages(tiny_tsv_path):
    client, model = make_client(tiny_tsv_path, lambda m: AIMessage(content="ok"))
    with client:
        cid = dict(parse_sse(chat(client, "one").text))["meta"]["conversation_id"]
        chat(client, "two", conversation_id=cid)
        chat(client, "three", conversation_id="someone-else")
    humans = [sum(isinstance(m, HumanMessage) for m in seen) for seen in model.seen]
    assert humans == [1, 2, 1]


def test_empty_or_oversized_message_is_422(tiny_tsv_path):
    client, _ = make_client(tiny_tsv_path, lambda m: AIMessage(content="ok"))
    with client:
        assert chat(client, "   ").status_code == 422
        assert chat(client, "x" * 2001).status_code == 422
        assert chat(client, "x" * 2000).status_code == 200


def test_model_failure_is_error_then_done(tiny_tsv_path):
    def boom(msgs):
        raise RuntimeError("provider exploded")

    client, _ = make_client(tiny_tsv_path, boom)
    with client:
        events = parse_sse(chat(client).text)
    assert [n for n, _ in events] == ["meta", "error", "done"]
    assert "provider exploded" in events[1][1]["message"]


def test_looping_model_is_hard_stop_error_then_done(tiny_tsv_path):
    n = {"i": 0}

    def loop(msgs):
        n["i"] += 1
        return call("count_neurons", {"group_by": "side"}, f"c{n['i']}")

    client, _ = make_client(tiny_tsv_path, loop, max_steps=2)
    with client:
        events = parse_sse(chat(client).text)
    names = [e for e, _ in events]
    assert names[-2:] == ["error", "done"]
    assert "hard stop" in events[-2][1]["message"]
    assert "answer" not in names


def test_no_model_configured_is_error_then_done(tiny_tsv_path):
    with TestClient(create_app(atlas_path=tiny_tsv_path)) as client:
        events = parse_sse(chat(client).text)
    assert [n for n, _ in events] == ["meta", "error", "done"]


def test_long_tool_args_are_truncated(tiny_tsv_path):
    def reply(msgs):
        if msgs[-1].type == "human":
            return call("search_neurons", {"cell_type": "A" * 150})
        return AIMessage(content="ok")

    client, _ = make_client(tiny_tsv_path, reply)
    with client:
        args = dict(parse_sse(chat(client).text))["tool_call"]["args"]
    assert len(args["cell_type"]) <= 101  # 100 chars + ellipsis


def test_unknown_selection_is_404_with_clear_message(tiny_tsv_path):
    client, _ = make_client(tiny_tsv_path, lambda m: AIMessage(content="ok"))
    with client:
        r = client.get("/selections/sel_nope00/indices")
    assert r.status_code == 404
    assert "unknown selection" in r.json()["detail"]


def test_cors_allows_vite_origin(tiny_tsv_path):
    client, _ = make_client(tiny_tsv_path, lambda m: AIMessage(content="ok"))
    with client:
        ok = client.get("/health", headers={"Origin": "http://localhost:5173"})
        bad = client.get("/health", headers={"Origin": "http://evil.example"})
    assert ok.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-origin" not in bad.headers


def test_cors_origins_from_env(tiny_tsv_path, monkeypatch):
    monkeypatch.setenv("FLYBRAIN_CORS_ORIGINS", "http://a.test, http://b.test")
    with TestClient(create_app(atlas_path=tiny_tsv_path)) as client:
        r = client.get("/health", headers={"Origin": "http://b.test"})
    assert r.headers["access-control-allow-origin"] == "http://b.test"
