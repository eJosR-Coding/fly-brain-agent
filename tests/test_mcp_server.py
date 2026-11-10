import asyncio
import json

from flybrain.mcp_server import build_server


def test_server_lists_the_four_tools(tiny_atlas):
    server = build_server(tiny_atlas)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"search_neurons", "count_neurons", "describe_neuron", "list_selection"}


def test_search_through_the_server_returns_a_selection_id(tiny_atlas):
    server = build_server(tiny_atlas)
    result = asyncio.run(server.call_tool("search_neurons", {"nt": "acetylcholine"}))
    payload = result[1] if isinstance(result, tuple) else result
    if isinstance(payload, list):  # content blocks only
        payload = json.loads(payload[0].text)
    assert payload["selection_id"].startswith("sel_")


# --- hardening (H2) ---------------------------------------------------------------

import subprocess
import sys

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from flybrain import mcp_server


def test_unknown_neuron_text_is_readable_over_mcp(tiny_atlas):
    server = build_server(tiny_atlas)
    with pytest.raises(ToolError) as exc:
        asyncio.run(server.call_tool("describe_neuron", {"root_id": 12345}))
    assert "no neuron with root_id 12345; get ids from search_neurons or list_selection" in str(exc.value)
    assert "KeyError" not in str(exc.value)


def test_expired_ticket_text_is_readable_over_mcp(tiny_atlas, monkeypatch):
    from flybrain.selections import SelectionStore

    store = SelectionStore(max_items=1)
    server = build_server(tiny_atlas, store)
    first = asyncio.run(server.call_tool("search_neurons", {"nt": "glutamate"}))
    payload = first[1] if isinstance(first, tuple) else first
    if isinstance(payload, list):
        payload = json.loads(payload[0].text)
    sid = payload["selection_id"]
    asyncio.run(server.call_tool("search_neurons", {"nt": "acetylcholine"}))  # evicts sid
    with pytest.raises(ToolError, match=r"expired.*search_neurons\(nt='glutamate'\)"):
        asyncio.run(server.call_tool("list_selection", {"selection_id": sid}))


def test_tool_docs_list_every_side_value_and_the_caps(tiny_atlas):
    tools_by_name = {t.name: t for t in asyncio.run(build_server(tiny_atlas).list_tools())}
    search_doc = tools_by_name["search_neurons"].description
    assert "center" in search_doc and "na" in search_doc.split("side:")[1]
    assert "empty" in search_doc.lower()
    assert "100" in tools_by_name["count_neurons"].description


def test_main_with_bad_annotations_is_one_stderr_line_exit_1(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("FLYBRAIN_ANNOTATIONS", str(tmp_path / "nope.tsv"))
    with pytest.raises(SystemExit) as exc:
        mcp_server.main()
    out = capsys.readouterr()
    assert exc.value.code == 1
    assert out.out == ""
    assert out.err.count("\n") == 1 and "not found" in out.err


def test_main_with_bad_selection_cap_is_one_stderr_line_exit_1(monkeypatch, capsys, tiny_atlas):
    monkeypatch.setattr(mcp_server, "load_atlas", lambda path: tiny_atlas)
    monkeypatch.setenv("FLYBRAIN_MAX_SELECTIONS", "zero")
    with pytest.raises(SystemExit) as exc:
        mcp_server.main()
    out = capsys.readouterr()
    assert exc.value.code == 1 and out.out == ""
    assert "FLYBRAIN_MAX_SELECTIONS" in out.err and out.err.count("\n") == 1


def test_server_process_dies_cleanly_on_bad_annotations(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "flybrain.mcp_server"],
        env={"FLYBRAIN_ANNOTATIONS": str(tmp_path / "nope.tsv"), "PATH": ""},
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 1 and proc.stdout == ""
    assert proc.stderr.strip().count("\n") == 0 and "Traceback" not in proc.stderr
