"""flybrain-web CLI: argument parsing and friendly failure, never a real model."""

import pytest

from flybrain import web


def test_defaults_bind_to_localhost(monkeypatch):
    monkeypatch.delenv("FLYBRAIN_MODEL", raising=False)
    args = web.parse_args([])
    assert (args.host, args.port) == ("127.0.0.1", 8000)


def test_model_comes_from_env(monkeypatch):
    monkeypatch.setenv("FLYBRAIN_MODEL", "openai:gpt-x")
    assert web.parse_args([]).model == "openai:gpt-x"


def test_help_needs_no_model_or_key(capsys):
    with pytest.raises(SystemExit) as exc:
        web.parse_args(["--help"])
    assert exc.value.code == 0
    assert "--port" in capsys.readouterr().out


def test_bad_model_setup_is_one_friendly_line(monkeypatch, capsys):
    def boom(spec):
        raise ValueError("no key")

    monkeypatch.setattr("langchain.chat_models.init_chat_model", boom)
    assert web.main(["--model", "openai:x"]) == 1
    err = capsys.readouterr().err
    assert "could not set up model 'openai:x': no key" in err


def test_bad_model_spec_is_rejected():
    with pytest.raises(SystemExit):
        web.parse_args(["--model", "nocolon"])


def test_bad_turn_timeout_env_is_one_friendly_line(monkeypatch, capsys):
    monkeypatch.setattr("langchain.chat_models.init_chat_model", lambda spec: object())
    monkeypatch.setenv("FLYBRAIN_TURN_TIMEOUT", "abc")
    assert web.main(["--model", "openai:x"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "FLYBRAIN_TURN_TIMEOUT" in err and "'abc'" in err


def test_models_flag_and_env(monkeypatch):
    monkeypatch.delenv("FLYBRAIN_MODELS", raising=False)
    assert web.parse_args([]).models is None
    assert web.parse_args(["--models", "groq:a, groq:b"]).models == "groq:a,groq:b"
    monkeypatch.setenv("FLYBRAIN_MODELS", "groq:c")
    assert web.parse_args([]).models == "groq:c"


def test_bad_models_list_is_rejected():
    with pytest.raises(SystemExit):
        web.parse_args(["--models", "groq:a,nocolon"])


def test_main_passes_the_resolved_registry_to_the_app(monkeypatch):
    seen = {}
    monkeypatch.setattr("langchain.chat_models.init_chat_model", lambda spec: object())
    monkeypatch.setattr("flybrain.api.create_app", lambda **kw: seen.update(kw) or object())
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    monkeypatch.delenv("FLYBRAIN_MODELS", raising=False)
    assert web.main(["--model", "google_genai:g", "--models", "groq:a"]) == 0
    assert seen["models"] == ["google_genai:g", "groq:a"]
