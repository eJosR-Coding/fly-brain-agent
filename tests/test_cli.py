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


# --- hardening (H3): each failure class has its own message -----------------------

import contextlib

import anyio
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from flybrain.agent_mcp import ServerStoppedError


@pytest.mark.parametrize("bad", ["gpt4", ":gpt", "openai:", " : ", ""])
def test_bad_model_shape_is_an_argparse_error(bad, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--model", bad, "hi"])
    assert exc.value.code == 2
    assert "provider:model" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["0", "-3", "abc", "1.5"])
def test_bad_max_steps_is_an_argparse_error(bad, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--max-steps", bad, "hi"])
    assert exc.value.code == 2
    assert "--max-steps" in capsys.readouterr().err


def test_max_steps_help_says_tool_rounds(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "tool rounds" in capsys.readouterr().out


@pytest.fixture
def wired(monkeypatch):
    """Fake model + fake MCP session + scripted ask: nothing real is ever reached."""
    state = {"ask": None, "enter": None, "init": None}

    def init(model):
        if state["init"]:
            raise state["init"]
        return object()

    @contextlib.asynccontextmanager
    async def fake_tools(path):
        if state["enter"]:
            raise state["enter"]
        yield []

    async def fake_ask(graph, question, max_steps):
        if isinstance(state["ask"], BaseException):
            raise state["ask"]
        return state["ask"]

    monkeypatch.setattr(cli, "init_chat_model", init)
    monkeypatch.setattr(cli, "mcp_tools", fake_tools)
    monkeypatch.setattr(cli, "build_graph", lambda model, tools: object())
    monkeypatch.setattr(cli, "ask", fake_ask)
    return state


def test_happy_path_prints_the_answer(wired, capsys):
    wired["ask"] = "5909 dopamine neurons"
    assert cli.main(["hi"]) == 0
    assert capsys.readouterr().out.strip() == "5909 dopamine neurons"


def test_model_setup_failure_names_model_and_hint(wired, capsys):
    wired["init"] = ImportError("no langchain_openai")
    assert cli.main(["--model", "openai:gpt-x", "hi"]) == 1
    err = capsys.readouterr().err
    assert "could not set up model 'openai:gpt-x': no langchain_openai" in err
    assert "API key" in err and "Traceback" not in err


def test_server_start_failure_does_not_blame_the_model(wired, capsys):
    wired["enter"] = ExceptionGroup(
        "tg", [McpError(ErrorData(code=-32000, message="Connection closed"))]
    )
    assert cli.main(["hi"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("flybrain-chat: flybrain server failed to start: Connection closed")
    assert "model" not in err.lower()


def test_server_stopped_mid_run_shows_its_own_message(wired, capsys):
    wired["ask"] = ServerStoppedError("the flybrain server stopped: Connection closed")
    assert cli.main(["hi"]) == 1
    err = capsys.readouterr().err
    assert "the flybrain server stopped: Connection closed" in err
    assert "could not set up model" not in err and "model call failed" not in err


def test_provider_runtime_error_is_model_call_failed(wired, capsys):
    wired["ask"] = RuntimeError("401 invalid api key\nrequest id 42")
    assert cli.main(["hi"]) == 1
    err = capsys.readouterr().err
    assert "model call failed: 401 invalid api key request id 42" in err
    assert err.count("\n") == 1  # one line


def test_unexpected_server_start_error_is_one_line(wired, capsys):
    wired["enter"] = OSError("python not found")
    assert cli.main(["hi"]) == 1
    assert "flybrain server failed to start: python not found" in capsys.readouterr().err


def test_real_server_failing_to_start_is_reported_as_server_failure(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "init_chat_model", lambda model: object())  # no real provider
    monkeypatch.setenv("FLYBRAIN_ANNOTATIONS", str(tmp_path / "nope.tsv"))
    assert cli.main(["hi"]) == 1
    err = capsys.readouterr().err
    assert "flybrain server failed to start" in err
    assert "model" not in err.split("flybrain server failed to start")[1].lower()
    assert "Traceback" not in err
