"""Model registry: the provider:model ids the server offers and whether each has a key.

Availability is a PRESENCE check on the provider's API key env var (the value is never read
beyond "is it non-empty"). Clients may only pick ids from this registry, never free text.
"""

from collections.abc import Mapping

# Groq models with tool use, from https://console.groq.com/docs/tool-use (production tier
# on https://console.groq.com/docs/models).
# Passed to init_chat_model for every chat model. Provider SDKs retry 429/5xx with backoff
# (Groq and Google default to 2 retries, honouring Retry-After), which on a rate-limited free
# tier looks like a hang. One retry, then the friendly quota message reaches the user.
# `max_retries` is the kwarg of langchain-groq, -google-genai, -anthropic and -openai alike.
CHAT_MODEL_KWARGS = {"max_retries": 1}

DEFAULT_EXTRA_MODELS = [
    # checked against the Groq models API (2026-11-10): the Llama 3 ids were no longer served
    "groq:openai/gpt-oss-120b",
    "groq:openai/gpt-oss-20b",
]

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


def resolve_models(default: str, raw: str | None) -> list[str]:
    """The registry: the default model first, then the configured list (or the built-in one)."""
    extra = parse_model_list(raw) if raw and raw.strip() else DEFAULT_EXTRA_MODELS
    return [default, *[m for m in extra if m != default]]


def describe_models(ids: list[str], env: Mapping[str, str]) -> list[dict]:
    """One row per id: {id, provider, available, reason?} from the provider key's presence."""
    rows = []
    for model_id in ids:
        provider = model_id.partition(":")[0]
        key = PROVIDER_KEYS.get(provider)
        if key is None:
            rows.append({"id": model_id, "provider": provider, "available": False,
                         "reason": f"unknown provider {provider!r}"})
        elif not env.get(key, "").strip():
            rows.append({"id": model_id, "provider": provider, "available": False,
                         "reason": f"{key} is not set"})
        else:
            rows.append({"id": model_id, "provider": provider, "available": True})
    return rows
