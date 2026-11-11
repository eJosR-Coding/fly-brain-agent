"""GET /models and the per-message model choice on POST /chat. Fake models, no network."""

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from flybrain.api import create_app
from fakes import fn_model
from test_api_chat import chat, parse_sse

ENV = {"GOOGLE_API_KEY": "x", "GROQ_API_KEY": "x"}  # no ANTHROPIC key


def answering(text):
    return fn_model(lambda m: AIMessage(content=text))


def make(path, **kw):
    built = {}

    def factory(model_id):
        built[model_id] = built.get(model_id, 0) + 1
        return answering(f"from {model_id}")

    default = answering("from default")
    app = create_app(
        atlas_path=path, model=default, model_name="google_genai:g",
        models=["google_genai:g", "groq:fast", "anthropic:claude"], model_factory=factory, env=ENV, **kw,
    )
    return TestClient(app), built, default


def test_models_lists_default_and_availability(tiny_tsv_path):
    client, _, _ = make(tiny_tsv_path)
    with client:
        body = client.get("/models").json()
    assert body["default"] == "google_genai:g"
    by_id = {m["id"]: m for m in body["models"]}
    assert list(by_id) == ["google_genai:g", "groq:fast", "anthropic:claude"]
    assert by_id["groq:fast"] == {"id": "groq:fast", "provider": "groq", "available": True}
    assert by_id["anthropic:claude"]["available"] is False
    assert "ANTHROPIC_API_KEY" in by_id["anthropic:claude"]["reason"]
    assert client.get("/health").json()["model"] == "google_genai:g"


def test_models_empty_without_a_chat_model(tiny_tsv_path):
    with TestClient(create_app(atlas_path=tiny_tsv_path)) as client:
        assert client.get("/models").json() == {"default": None, "models": []}


def test_chat_without_model_uses_default_and_meta_names_it(tiny_tsv_path):
    client, built, _ = make(tiny_tsv_path)
    with client:
        events = parse_sse(chat(client, "hi").text)
    assert events[0][1]["model"] == "google_genai:g"
    assert dict(events)["answer"] == {"text": "from default"}
    assert built == {}


def test_chat_with_model_builds_it_lazily_once_and_reports_it(tiny_tsv_path):
    client, built, _ = make(tiny_tsv_path)
    with client:
        assert built == {}
        first = parse_sse(chat(client, "hi", model="groq:fast").text)
        parse_sse(chat(client, "again", model="groq:fast").text)
    assert first[0][1]["model"] == "groq:fast"
    assert dict(first)["answer"] == {"text": "from groq:fast"}
    assert built == {"groq:fast": 1}


@pytest.mark.parametrize("bad", ["groq:other", "anthropic:claude", "openai:gpt; rm -rf", ""])
def test_chat_rejects_unknown_or_unavailable_models_with_422(tiny_tsv_path, bad):
    client, built, _ = make(tiny_tsv_path)
    with client:
        r = chat(client, "hi", model=bad)
    assert r.status_code == 422
    assert r.json()["detail"]["available"] == ["google_genai:g", "groq:fast"]
    assert built == {}  # a client string never reaches the factory


def test_switching_model_keeps_the_conversation_history(tiny_tsv_path):
    seen = {}

    def factory(model_id):
        m = fn_model(lambda msgs: AIMessage(content="ok"))
        seen[model_id] = m
        return m

    default = fn_model(lambda msgs: AIMessage(content="ok"))
    app = create_app(atlas_path=tiny_tsv_path, model=default, model_name="google_genai:g",
                     models=["google_genai:g", "groq:fast"], model_factory=factory, env=ENV)
    with TestClient(app) as client:
        cid = dict(parse_sse(chat(client, "one").text))["meta"]["conversation_id"]
        chat(client, "two", conversation_id=cid, model="groq:fast")
    humans = [m.content for m in seen["groq:fast"].seen[0] if isinstance(m, HumanMessage)]
    assert humans == ["one", "two"]


def test_factory_failure_becomes_an_error_event(tiny_tsv_path):
    def factory(model_id):
        raise ImportError("langchain-groq is not installed")

    app = create_app(atlas_path=tiny_tsv_path, model=answering("x"), model_name="google_genai:g",
                     models=["google_genai:g", "groq:fast"], model_factory=factory, env=ENV)
    with TestClient(app) as client:
        events = parse_sse(chat(client, "hi", model="groq:fast").text)
    names = [n for n, _ in events]
    assert names == ["meta", "error", "done"]
    assert "groq:fast" in dict(events)["error"]["message"]


def test_provider_errors_name_the_chosen_model(tiny_tsv_path):
    class Boom(Exception):
        status_code = 429

    def factory(model_id):
        def reply(msgs):
            raise Boom("429 RESOURCE_EXHAUSTED")

        return fn_model(reply)

    app = create_app(atlas_path=tiny_tsv_path, model=answering("x"), model_name="google_genai:g",
                     models=["google_genai:g", "groq:fast"], model_factory=factory, env=ENV)
    with TestClient(app) as client:
        events = parse_sse(chat(client, "hi", model="groq:fast").text)
    assert dict(events)["error"]["message"].startswith("model groq:fast hit its quota or rate limit")


def test_models_picked_in_the_picker_are_built_to_fail_fast(tiny_tsv_path, monkeypatch):
    seen = []

    def init(spec, **kwargs):
        seen.append((spec, kwargs))
        return answering("from picker")

    monkeypatch.setattr("langchain.chat_models.init_chat_model", init)
    app = create_app(atlas_path=tiny_tsv_path, model=answering("from default"), model_name="google_genai:g",
                     models=["google_genai:g", "groq:fast"], env=ENV)  # default factory
    with TestClient(app) as client:
        events = parse_sse(chat(client, "hi", model="groq:fast").text)
    assert dict(events)["answer"] == {"text": "from picker"}
    assert seen == [("groq:fast", {"max_retries": 1})]


def test_models_hides_ids_the_provider_does_not_serve(tiny_tsv_path):
    app = create_app(
        atlas_path=tiny_tsv_path, model=answering("x"), model_name="google_genai:g",
        models=["google_genai:g", "groq:old", "groq:new"], env=ENV,
        served={"groq": {"new"}, "google_genai": {"g"}},
    )
    with TestClient(app) as client:
        by_id = {m["id"]: m for m in client.get("/models").json()["models"]}
    assert by_id["groq:new"]["available"] is True
    assert by_id["groq:old"] == {"id": "groq:old", "provider": "groq", "available": False,
                                 "reason": "not served for this key"}
