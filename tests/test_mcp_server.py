import asyncio
import json

from flybrain.mcp_server import build_server


def test_server_lists_the_tools(tiny_atlas):
    server = build_server(tiny_atlas)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {
        "search_neurons", "count_neurons", "describe_neuron", "list_selection",
        "neighbors", "find_path", "lesion_effect", "top_hubs", "preset_selection", "stimulate",
        "dose_response", "silencing_screen",
    }


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


# --- connectome tools --------------------------------------------------------------

def _call(server, name, args):
    result = asyncio.run(server.call_tool(name, args))
    payload = result[1] if isinstance(result, tuple) else result
    if isinstance(payload, list):
        payload = json.loads(payload[0].text)
    return payload


def test_connectome_tools_without_a_connectome_give_the_guiding_error(tiny_atlas):
    server = build_server(tiny_atlas)  # no connectome: the old tools must keep working
    sid = _call(server, "search_neurons", {"cell_type": "EPG"})["selection_id"]
    with pytest.raises(ToolError, match="flybrain-build-connectome"):
        asyncio.run(server.call_tool("neighbors", {"selection_id": sid}))
    with pytest.raises(ToolError, match="flybrain-build-connectome"):
        asyncio.run(server.call_tool("top_hubs", {}))


def test_neighbors_through_the_server_with_a_connectome(tiny_atlas, tiny_conn):
    server = build_server(tiny_atlas, conn=tiny_conn)
    sid = _call(server, "search_neurons", {"cell_type": "EPG"})["selection_id"]
    out = _call(server, "neighbors", {"selection_id": sid, "direction": "out"})
    assert out["top"][0]["root_id"] == 1 and out["selection_id"].startswith("sel_")


def test_connectome_tool_docs_explain_the_threshold_and_honesty(tiny_atlas):
    by_name = {t.name: t for t in asyncio.run(build_server(tiny_atlas).list_tools())}
    for name in ("neighbors", "find_path", "lesion_effect", "top_hubs"):
        doc = by_name[name].description.lower()
        assert "structural" in doc, name
    assert "5 synapses" in by_name["neighbors"].description
    assert "directed" in by_name["find_path"].description.lower()


def test_load_connectome_if_present_is_none_when_missing(tiny_atlas, tmp_path):
    assert mcp_server.load_connectome(tiny_atlas, tmp_path / "nope.npz") is None


def test_load_connectome_if_present_skips_a_table_for_another_atlas(tiny_atlas, tmp_path):
    import numpy as np

    from flybrain.connectome_build import write_npz

    path = tmp_path / "c.npz"
    e = np.array([0], dtype=np.int32)
    write_npz(path, e, e, e, n_neurons=7, source="x", stats={"rows_in": 1, "dropped_unknown": 0, "dropped_self_loops": 0})
    assert mcp_server.load_connectome(tiny_atlas, path) is None


def test_find_path_doc_explains_strength_fields(tiny_atlas):
    by_name = {t.name: t for t in asyncio.run(build_server(tiny_atlas).list_tools())}
    doc = by_name["find_path"].description
    for field in ("synapses_from_previous", "min_hop_synapses", "total_synapses",
                  "shortest_path_count", "alternative_routes", "bottleneck"):
        assert field in doc, field


# --- token budget: the tool definitions ride on EVERY model call ---------------------------------

TOOL_DEFS_BUDGET_CHARS = 8400  # ~2,100 tokens; was 12,320 chars (~3,100 tokens) before trimming; +2 experiments


def _definition_chars(tiny_atlas) -> dict[str, int]:
    tools_ = asyncio.run(build_server(tiny_atlas).list_tools())
    return {
        t.name: len(json.dumps({"name": t.name, "description": t.description, "inputSchema": t.inputSchema}))
        for t in tools_
    }


def test_tool_definitions_stay_within_the_token_budget(tiny_atlas):
    sizes = _definition_chars(tiny_atlas)
    assert len(sizes) == 12
    assert sum(sizes.values()) <= TOOL_DEFS_BUDGET_CHARS, sizes


def test_concise_docs_keep_the_interpretation_rules(tiny_atlas):
    by_name = {t.name: t for t in asyncio.run(build_server(tiny_atlas).list_tools())}
    assert "not the living fly" in by_name["stimulate"].description
    assert "silence_selection_id" in by_name["stimulate"].inputSchema["properties"]
    assert "top_non_sensory" in by_name["stimulate"].description  # where silencing candidates come from
    assert "25" in by_name["list_selection"].description
    assert "selection_id" in by_name["search_neurons"].description  # the ticket concept
    assert "not signal flow" in by_name["neighbors"].description
    assert "reachab" in by_name["lesion_effect"].description.lower()
    assert "sugar" in by_name["preset_selection"].description and "mn9" in by_name["preset_selection"].description


def test_schemas_carry_no_title_noise(tiny_atlas):
    for t in asyncio.run(build_server(tiny_atlas).list_tools()):
        assert "title" not in t.inputSchema
        assert all("title" not in p for p in t.inputSchema["properties"].values()), t.name
