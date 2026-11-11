"""Provider errors become one readable line that names the model and the fix."""

from flybrain.agent import friendly_model_error
from flybrain.models import DEFAULT_EXTRA_MODELS


class NotFound(Exception):
    status_code = 404


def test_unknown_model_for_this_key_says_pick_another():
    exc = NotFound(
        "Error code: 404 - {'error': {'message': 'The model `llama-3.3-70b-versatile` does not exist "
        "or you do not have access to it.', 'type': 'invalid_request_error', 'code': 'model_not_found'}}"
    )
    msg = friendly_model_error(exc, "groq:llama-3.3-70b-versatile")
    assert msg == (
        "model groq:llama-3.3-70b-versatile is not available for this API key; "
        "pick another model in the picker"
    )


def test_model_not_found_is_detected_from_the_text_alone():
    msg = friendly_model_error(Exception("code: 'model_not_found'"), "groq:x")
    assert msg is not None and "not available for this API key" in msg


def test_a_plain_404_about_something_else_is_not_mislabeled():
    assert friendly_model_error(NotFound("route /v1/foo not found"), "groq:x") is None


def test_default_groq_models_are_the_ones_groq_still_serves():
    # verified 2026-11-10 with the Groq models API on the user's key: the Llama 3 ids were gone
    assert DEFAULT_EXTRA_MODELS == ["groq:openai/gpt-oss-120b", "groq:openai/gpt-oss-20b"]


class TooLarge(Exception):
    status_code = 413


GROQ_413 = (
    "Error code: 413 - {'error': {'message': 'Request too large for model `openai/gpt-oss-120b` in "
    "organization `org_x` service tier `on_demand` on tokens per minute (TPM): Limit 8000, Requested "
    "11874, please reduce your message size and try again.', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)
TOO_LARGE_MSG = (
    "model groq:openai/gpt-oss-120b: request too large for its per-minute token limit; "
    "start a new chat or ask a narrower question"
)


def test_413_is_named_request_too_large_not_quota():
    assert friendly_model_error(TooLarge(GROQ_413), "groq:openai/gpt-oss-120b") == TOO_LARGE_MSG


def test_request_too_large_text_alone_is_detected_despite_the_rate_limit_code():
    # the Groq body also says rate_limit_exceeded; the size wording must win over the 429 wording
    assert friendly_model_error(Exception(GROQ_413), "groq:openai/gpt-oss-120b") == TOO_LARGE_MSG
    assert friendly_model_error(Exception("request too large"), "groq:openai/gpt-oss-120b") == TOO_LARGE_MSG


def test_a_real_429_is_still_a_quota_message():
    class Limited(Exception):
        status_code = 429

    msg = friendly_model_error(Limited("Rate limit reached for model x"), "groq:x")
    assert msg is not None and "quota or rate limit" in msg and "too large" not in msg
