"""Model registry: the provider:model ids the server offers and whether each has a key.

Availability is a PRESENCE check on the provider's API key env var (the value is never read
beyond "is it non-empty"). Clients may only pick ids from this registry, never free text.
"""

from collections.abc import Callable, Mapping

import httpx

# Passed to init_chat_model for every chat model. Provider SDKs retry 429/5xx with backoff
# (Groq and Google default to 2 retries, honouring Retry-After), which on a rate-limited free
# tier looks like a hang. One retry, then the friendly quota message reaches the user.
# `max_retries` is the kwarg of langchain-groq, -google-genai, -anthropic and -openai alike.
CHAT_MODEL_KWARGS = {"max_retries": 1}

# Curated tool-calling chat models, cheapest first (provider order: deepseek, groq, google,
# anthropic). Only ids verified against each provider's docs or models API. A model shows in
# the picker when its provider key is set and the provider still serves it (see discover_served).
# openai: none yet, no verified tool-calling id; use --models for it.
CATALOG = [
    "deepseek:deepseek-flash",
    "deepseek:deepseek-v4-pro",
    "groq:openai/gpt-oss-120b",  # the Groq Llama 3 ids were retired (checked 2026-11-10)
    "groq:openai/gpt-oss-20b",
    "google_genai:gemini-3.5-flash",
    "anthropic:claude-haiku-5-5",
    "anthropic:claude-sonnet-5-5",
]

LIST_TIMEOUT = 5.0  # seconds per provider list-models request

PROVIDER_KEYS = {
    "google_genai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _split(model_id: str) -> tuple[str, str]:
    provider, sep, name = model_id.partition(":")
    if not sep or not provider.strip() or not name.strip():
        raise ValueError(f"{model_id!r} is not a valid model; use provider:model")
    return provider.strip(), name.strip()


def parse_model_list(raw: str) -> list[str]:
    """Comma-separated provider:model ids -> trimmed, de-duplicated, order kept."""
    ids: list[str] = []
    for part in raw.split(","):
        if not part.strip():
            continue
        provider, name = _split(part)
        model_id = f"{provider}:{name}"
        if model_id not in ids:
            ids.append(model_id)
    return ids


def resolve_models(default: str, raw: str | None, catalog: list[str] = ()) -> list[str]:
    """The registry: the default first, then the configured list (or the given catalog ids)."""
    extra = parse_model_list(raw) if raw and raw.strip() else list(catalog)
    return [default, *[m for m in extra if m != default]]


# provider -> (list-models url, headers built from the key)
def _bearer(key):
    return {"Authorization": f"Bearer {key}"}


_LIST_ENDPOINTS = {
    "groq": ("https://api.groq.com/openai/v1/models", _bearer),
    "deepseek": ("https://api.deepseek.com/models", _bearer),
    "openai": ("https://api.openai.com/v1/models", _bearer),
    "anthropic": ("https://api.anthropic.com/v1/models",
                  lambda key: {"x-api-key": key, "anthropic-version": "2023-06-01"}),
    "google_genai": ("https://generativelanguage.googleapis.com/v1beta/models",
                     lambda key: {"x-goog-api-key": key}),
}


def fetch_served(provider: str, key: str, client: httpx.Client | None = None) -> set[str]:
    """Model ids the provider serves for this key. Raises on any failure (the caller fails open).

    The key goes only in a header to the provider's own endpoint; errors name the status, never
    the key."""
    url, headers = _LIST_ENDPOINTS[provider]
    owns = client is None
    client = client or httpx.Client(timeout=LIST_TIMEOUT)
    try:
        resp = client.get(url, headers=headers(key), timeout=LIST_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"{provider} model list returned HTTP {exc.response.status_code}") from None
    except Exception as exc:
        raise RuntimeError(f"{provider} model list failed: {type(exc).__name__}") from None
    finally:
        if owns:
            client.close()
    if provider == "google_genai":  # names come back as "models/<id>"
        return {m["name"].removeprefix("models/") for m in body.get("models", [])}
    return {m["id"] for m in body.get("data", [])}


def discover_served(env: Mapping[str, str], fetcher: Callable[[str, str], set[str]] = fetch_served
                    ) -> dict[str, set[str] | None]:
    """provider -> served ids, for providers with a key; None when the fetch failed (fail open)."""
    served: dict[str, set[str] | None] = {}
    for provider, key_name in PROVIDER_KEYS.items():
        key = env.get(key_name, "").strip()
        if not key:
            continue
        try:
            served[provider] = fetcher(provider, key)
        except Exception:
            served[provider] = None
    return served


def startup_registry(explicit: str | None, raw: str | None, env: Mapping[str, str],
                     fetcher: Callable[[str, str], set[str]] = fetch_served):
    """(default id or None, registry ids, served map), computed once at startup.

    An explicit model wins as default; otherwise it is the first available id. `raw` (the
    --models list) replaces the catalog."""
    served = discover_served(env, fetcher)
    pool = parse_model_list(raw) if raw and raw.strip() else CATALOG
    available = [r["id"] for r in describe_models(pool, env, served) if r["available"]]
    default = explicit or (available[0] if available else None)
    if default is None:
        return None, [], served
    return default, resolve_models(default, raw, available), served


def describe_models(ids: list[str], env: Mapping[str, str],
                    served: Mapping[str, set[str] | None] | None = None) -> list[dict]:
    """One row per id: {id, provider, available, reason?, note?}.

    A key must be set. If `served` has a list for the provider, the model must be in it; a
    failed fetch (None) fails open with a note."""
    rows = []
    for model_id in ids:
        provider, _, name = model_id.partition(":")
        key = PROVIDER_KEYS.get(provider)
        if key is None:
            rows.append({"id": model_id, "provider": provider, "available": False,
                         "reason": f"unknown provider {provider!r}"})
        elif not env.get(key, "").strip():
            rows.append({"id": model_id, "provider": provider, "available": False,
                         "reason": f"{key} is not set"})
        elif served and served.get(provider) is not None and name not in served[provider]:
            rows.append({"id": model_id, "provider": provider, "available": False,
                         "reason": "not served for this key"})
        else:
            row = {"id": model_id, "provider": provider, "available": True}
            if served and provider in served and served[provider] is None:
                row["note"] = "model list unavailable; not verified"
            rows.append(row)
    return rows
