"""Hybrid catalog: curated ids, shown per key, cross-checked against the provider's list."""

import httpx
import pytest

from flybrain.models import (
    CATALOG,
    describe_models,
    discover_served,
    fetch_served,
    startup_registry,
)

ALL_KEYS = {
    "DEEPSEEK_API_KEY": "k1", "GROQ_API_KEY": "k2", "GOOGLE_API_KEY": "k3",
    "ANTHROPIC_API_KEY": "k4", "OPENAI_API_KEY": "k5",
}


def test_catalog_is_cheapest_first_and_curated():
    assert CATALOG == [
        "deepseek:deepseek-flash", "deepseek:deepseek-v4-pro",
        "groq:openai/gpt-oss-120b", "groq:openai/gpt-oss-20b",
        "google_genai:gemini-3.5-flash",
        "anthropic:claude-haiku-5-5", "anthropic:claude-sonnet-5-5",
    ]


def test_discover_skips_providers_without_a_key_and_passes_the_key_only_to_the_fetcher():
    calls = []

    def fetcher(provider, key):
        calls.append((provider, key))
        return {"x"}

    served = discover_served({"GROQ_API_KEY": "secret"}, fetcher)
    assert calls == [("groq", "secret")]
    assert served == {"groq": {"x"}}


def test_discover_records_none_when_a_fetch_fails_and_never_raises():
    def fetcher(provider, key):
        raise RuntimeError("boom secret")

    assert discover_served({"GROQ_API_KEY": "secret"}, fetcher) == {"groq": None}


def test_describe_hides_ids_missing_from_a_successful_list():
    served = {"groq": {"openai/gpt-oss-120b"}}
    rows = describe_models(
        ["groq:openai/gpt-oss-120b", "groq:openai/gpt-oss-20b"], {"GROQ_API_KEY": "k"}, served)
    assert rows[0]["available"] is True
    assert rows[1] == {"id": "groq:openai/gpt-oss-20b", "provider": "groq", "available": False,
                       "reason": "not served for this key"}


def test_describe_fails_open_when_the_list_could_not_be_fetched():
    rows = describe_models(["groq:a"], {"GROQ_API_KEY": "k"}, {"groq": None})
    assert rows[0]["available"] is True
    assert "not verified" in rows[0]["note"]


def test_describe_without_discovery_is_the_presence_check_as_before():
    rows = describe_models(["groq:a"], {"GROQ_API_KEY": "k"})
    assert rows == [{"id": "groq:a", "provider": "groq", "available": True}]


def test_registry_shows_only_catalog_models_whose_key_is_set():
    default, models, _ = startup_registry(None, None, {"GROQ_API_KEY": "k"}, lambda p, k: None_set())
    assert models == ["groq:openai/gpt-oss-120b", "groq:openai/gpt-oss-20b"]
    assert default == "groq:openai/gpt-oss-120b"


def None_set():
    raise OSError("offline")


def test_registry_drops_retired_ids_and_default_is_first_available():
    def fetcher(provider, key):
        return {"deepseek-v4-pro"} if provider == "deepseek" else {"openai/gpt-oss-20b"}

    env = {"DEEPSEEK_API_KEY": "a", "GROQ_API_KEY": "b"}
    default, models, served = startup_registry(None, None, env, fetcher)
    assert default == "deepseek:deepseek-v4-pro"
    assert models == ["deepseek:deepseek-v4-pro", "groq:openai/gpt-oss-20b"]
    assert served["groq"] == {"openai/gpt-oss-20b"}


def test_explicit_model_wins_and_stays_first():
    env = {"GROQ_API_KEY": "b"}
    default, models, _ = startup_registry("anthropic:claude-sonnet-5-5", None, env, lambda p, k: None_set())
    assert default == "anthropic:claude-sonnet-5-5"
    assert models[0] == "anthropic:claude-sonnet-5-5"
    assert "groq:openai/gpt-oss-120b" in models


def test_explicit_models_list_overrides_the_catalog():
    env = {"GROQ_API_KEY": "b", "DEEPSEEK_API_KEY": "a"}
    default, models, _ = startup_registry(None, "groq:openai/gpt-oss-20b", env, lambda p, k: {"openai/gpt-oss-20b"})
    assert models == ["groq:openai/gpt-oss-20b"]
    assert default == "groq:openai/gpt-oss-20b"


def test_no_keys_means_no_default_and_an_empty_registry():
    default, models, served = startup_registry(None, None, {}, lambda p, k: {"x"})
    assert (default, models, served) == (None, [], {})


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_served_groq_uses_bearer_and_reads_data_ids():
    seen = {}

    def handler(request):
        seen["url"], seen["auth"] = str(request.url), request.headers["authorization"]
        return httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}]})

    assert fetch_served("groq", "K", _client(handler)) == {"a", "b"}
    assert seen == {"url": "https://api.groq.com/openai/v1/models", "auth": "Bearer K"}


def test_fetch_served_anthropic_headers():
    def handler(request):
        assert request.headers["x-api-key"] == "K"
        assert request.headers["anthropic-version"] == "2023-06-01"
        return httpx.Response(200, json={"data": [{"id": "claude-haiku-5-5"}]})

    assert fetch_served("anthropic", "K", _client(handler)) == {"claude-haiku-5-5"}


def test_fetch_served_google_uses_header_not_query_and_strips_models_prefix():
    def handler(request):
        assert request.headers["x-goog-api-key"] == "K"
        assert "key" not in request.url.params
        return httpx.Response(200, json={"models": [{"name": "models/gemini-3.5-flash"}]})

    assert fetch_served("google_genai", "K", _client(handler)) == {"gemini-3.5-flash"}


def test_fetch_served_raises_on_http_error_without_leaking_the_key():
    def handler(request):
        return httpx.Response(401, json={"error": "bad"})

    try:
        fetch_served("deepseek", "SECRETKEY", _client(handler))
    except Exception as exc:
        assert "SECRETKEY" not in str(exc)
    else:
        raise AssertionError("expected an error")


def test_fetch_served_google_follows_page_tokens():
    # Google pages at 50 by default: a catalog model on page 2 must not look retired
    pages = {None: (["models/a"], "t2"), "t2": (["models/gemini-3.5-flash"], None)}

    def handler(request):
        names, nxt = pages[request.url.params.get("pageToken")]
        body = {"models": [{"name": n} for n in names]}
        if nxt:
            body["nextPageToken"] = nxt
        return httpx.Response(200, json=body)

    assert fetch_served("google_genai", "K", _client(handler)) == {"a", "gemini-3.5-flash"}


def test_fetch_served_anthropic_follows_has_more():
    def handler(request):
        if request.url.params.get("after_id") == "m1":
            return httpx.Response(200, json={"data": [{"id": "claude-haiku-5-5"}], "has_more": False})
        return httpx.Response(200, json={"data": [{"id": "m1"}], "has_more": True, "last_id": "m1"})

    assert fetch_served("anthropic", "K", _client(handler)) == {"m1", "claude-haiku-5-5"}


def test_fetch_served_gives_up_on_endless_pages_so_the_caller_fails_open():
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "models/x"}], "nextPageToken": "again"})

    with pytest.raises(RuntimeError, match="too many pages"):
        fetch_served("google_genai", "K", _client(handler))
