import pytest

from flybrain import cli


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "--model" in capsys.readouterr().out


def test_missing_provider_is_friendly(monkeypatch, capsys):
    def boom(model):
        raise ImportError("Unable to import langchain_openai")

    monkeypatch.setattr(cli, "init_chat_model", boom)
    code = cli.main(["--model", "openai:gpt-x", "hi"])
    err = capsys.readouterr().err
    assert code == 1
    assert "openai:gpt-x" in err
    assert "Traceback" not in err
