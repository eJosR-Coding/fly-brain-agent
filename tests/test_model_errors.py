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
