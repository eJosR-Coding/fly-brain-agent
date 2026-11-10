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
    assert payload["selection_id"] == "sel_1"
