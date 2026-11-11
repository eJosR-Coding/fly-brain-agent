"""The `path` SSE event: a find_path result becomes atlas rows + per-hop synapses for the viewer."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from conftest import TINY_EDGES
from flybrain.api import create_app, path_payload
from flybrain.atlas import RowIndex
from flybrain.connectome_build import write_npz
from fakes import call, fn_model
from test_api_chat import chat, parse_sse


@pytest.fixture
def conn_path(tmp_path):
    pre, post, syn = (np.array(c, dtype=np.int32) for c in zip(*TINY_EDGES))
    path = tmp_path / "c.npz"
    write_npz(path, pre, post, syn, n_neurons=63, source="x",
              stats={"rows_in": 1, "dropped_unknown": 0, "dropped_self_loops": 0})
    return path


def _tickets(msgs):
    return [json.loads(m.text)["selection_id"] for m in msgs if m.type == "tool" and m.name == "search_neurons"]


def path_script(frm, to):
    """search `frm`, search `to`, find_path between them, then answer."""

    def reply(msgs):
        done = sum(1 for m in msgs if m.type == "tool")
        if done == 0:
            return call("search_neurons", {"cell_type": frm}, "c1")
        if done == 1:
            return call("search_neurons", {"cell_type": to}, "c2")
        if done == 2:
            a, b = _tickets(msgs)
            return call("find_path", {"from_selection": a, "to_selection": b}, "c3")
        return AIMessage(content="done")

    return reply


def run(tiny_tsv_path, conn_path, reply):
    client = TestClient(create_app(atlas_path=tiny_tsv_path, connectome_path=conn_path, model=fn_model(reply)))
    with client:
        return parse_sse(chat(client, "path?").text)


def test_find_path_emits_a_path_event_after_its_highlight(tiny_tsv_path, conn_path):
    events = run(tiny_tsv_path, conn_path, path_script("ORN_DA1", "PEN"))
    names = [n for n, _ in events]
    i = names.index("path")
    assert names[i - 1] == "highlight" and names[i - 2] == "tool_result"
    path = events[i][1]
    assert path["rows"] == [62, 61, 60, 0]  # ORN_DA1 -> T4a -> EPG -> PEN_a, atlas rows in order
    assert path["synapses"] == [9, 6, 10]
    assert path["bottleneck"] == 6
    assert path["selection_id"] == events[i - 1][1]["selection_id"]


def test_other_tools_do_not_emit_path(tiny_tsv_path, conn_path):
    def reply(msgs):
        return call("search_neurons", {"nt": "glutamate"}) if msgs[-1].type == "human" else AIMessage(content="x")

    assert "path" not in [n for n, _ in run(tiny_tsv_path, conn_path, reply)]


def test_a_search_without_a_path_emits_nothing(tiny_tsv_path, conn_path):
    # PEN -> ORN_DA1 runs against the arrows: found is false
    names = [n for n, _ in run(tiny_tsv_path, conn_path, path_script("PEN", "ORN_DA1"))]
    assert "path" not in names and names[-1] == "done"


def test_path_payload_maps_root_ids_and_survives_missing_ones(tiny_atlas):
    idx = RowIndex(tiny_atlas)
    data = {"found": True, "selection_id": "sel_x", "min_hop_synapses": 5,
            "path": [{"root_id": 102, "synapses_from_previous": None},
                     {"root_id": 101, "synapses_from_previous": 9}]}
    assert path_payload(data, idx) == {"selection_id": "sel_x", "rows": [62, 61], "synapses": [9], "bottleneck": 5}
    gone = {**data, "path": [{"root_id": 102, "synapses_from_previous": None},
                             {"root_id": 424242, "synapses_from_previous": 9}]}
    assert path_payload(gone, idx) is None  # unknown neuron: no cable rather than a broken one


@pytest.mark.parametrize("data", [
    {"found": False}, {"found": True, "path": []}, {"found": True, "path": "x"},
    {"found": True, "path": [{"root_id": 102, "synapses_from_previous": None}]},  # 0 hops: nothing to draw
    [], None,
])
def test_path_payload_rejects_what_is_not_a_drawable_path(tiny_atlas, data):
    assert path_payload(data, RowIndex(tiny_atlas)) is None
