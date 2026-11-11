"""DeepSeek is a first-class provider: its availability follows DEEPSEEK_API_KEY."""

from flybrain.models import PROVIDER_KEYS


def test_deepseek_key_env_var_is_known():
    assert PROVIDER_KEYS["deepseek"] == "DEEPSEEK_API_KEY"


def test_deepseek_chat_model_is_buildable_without_network(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    from langchain.chat_models import init_chat_model

    model = init_chat_model("deepseek:deepseek-flash", max_retries=1)
    assert model.model_name == "deepseek-flash"
