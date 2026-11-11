"""Model registry: which provider:model ids exist and which of them have an API key set."""

import pytest

from flybrain.models import (
    DEFAULT_EXTRA_MODELS,
    describe_models,
    parse_model_list,
    resolve_models,
)


def test_parse_model_list_splits_trims_and_dedups():
    assert parse_model_list(" groq:a , google_genai:b,groq:a,, ") == ["groq:a", "google_genai:b"]


def test_parse_model_list_keeps_slashes_in_the_model_part():
    assert parse_model_list("groq:openai/gpt-oss-120b") == ["groq:openai/gpt-oss-120b"]


@pytest.mark.parametrize("bad", ["nocolon", "groq:", ":model", "groq:a,oops"])
def test_parse_model_list_rejects_malformed_ids(bad):
    with pytest.raises(ValueError):
        parse_model_list(bad)


def test_resolve_default_list_has_current_model_first_then_groq_defaults():
    ids = resolve_models("google_genai:gemini-3.5-flash", None)
    assert ids[0] == "google_genai:gemini-3.5-flash"
    assert ids[1:] == DEFAULT_EXTRA_MODELS
    assert all(m.startswith("groq:") for m in DEFAULT_EXTRA_MODELS)


def test_resolve_explicit_list_always_includes_the_default_first():
    assert resolve_models("anthropic:x", "groq:a,anthropic:x,groq:b") == ["anthropic:x", "groq:a", "groq:b"]


def test_describe_marks_models_without_a_key_as_unavailable():
    env = {"GROQ_API_KEY": "dummy"}
    rows = describe_models(["groq:a", "google_genai:b", "anthropic:c", "openai:d", "mystery:e"], env)
    by_id = {r["id"]: r for r in rows}
    assert by_id["groq:a"] == {"id": "groq:a", "provider": "groq", "available": True}
    assert by_id["google_genai:b"]["available"] is False
    assert by_id["google_genai:b"]["reason"] == "GOOGLE_API_KEY is not set"
    assert by_id["anthropic:c"]["reason"] == "ANTHROPIC_API_KEY is not set"
    assert by_id["openai:d"]["reason"] == "OPENAI_API_KEY is not set"
    assert by_id["mystery:e"]["available"] is False  # unknown provider: never offered


def test_empty_key_counts_as_not_set():
    rows = describe_models(["groq:a"], {"GROQ_API_KEY": "  "})
    assert rows[0]["available"] is False
