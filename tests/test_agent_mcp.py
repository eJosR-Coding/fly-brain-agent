"""Agent <-> real flybrain MCP server over stdio. Fake model, no network, no API key."""

import asyncio
import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from flybrain.agent import ask, build_graph
from flybrain.agent_mcp import mcp_tools

HEADER = "root_id\tpos_x\tpos_y\tpos_z\tsuper_class\tcell_class\tcell_type\ttop_nt\tside\tflow"


class FnModel(GenericFakeChatModel):
    """Fake model whose reply is a function of the conversation so far."""

    reply: object = None

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        return ChatResult(generations=[ChatGeneration(message=self.reply(messages))])


def tiny_tsv(tmp_path):
    rows = [f"{i}\t1000\t2000\t100\tcentral\tCX\tPEN_a\tacetylcholine\tleft\tintrinsic" for i in range(1, 4)]
    rows += ["10\t3000\t2500\t200\tcentral\tCX\tEPG\tglutamate\tright\tintrinsic"]
    path = tmp_path / "ann.tsv"
    path.write_text("\n".join([HEADER, *rows]) + "\n")
    return path


def call(name, args, cid):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])


def last_tool_json(messages):
    msg = messages[-1]
    assert isinstance(msg, ToolMessage)
    assert msg.status != "error", msg.content
    text = msg.content if isinstance(msg.content, str) else msg.content[0]["text"]
    return json.loads(text)


def test_count_neurons_through_real_server(tmp_path):
    seen = {}

    def reply(messages):
        if messages[-1].type == "human":
            return call("count_neurons", {"group_by": "top_nt"}, "c1")
        seen.update(last_tool_json(messages))
        return AIMessage(content="3 ACh, 1 Glu")

    async def run():
        async with mcp_tools(tiny_tsv(tmp_path)) as tools:
            graph = build_graph(FnModel(messages=iter([]), reply=reply), tools)
            return await ask(graph, "NT counts?")

    assert asyncio.run(run()) == "3 ACh, 1 Glu"
    assert '"acetylcholine": 3' in json.dumps(seen)


def test_ticket_survives_across_tool_calls(tmp_path):
    listed = {}

    def reply(messages):
        if messages[-1].type == "human":
            return call("search_neurons", {"cell_type": "PEN"}, "c1")
        data = last_tool_json(messages)
        if messages[-1].name == "search_neurons":
            return call("list_selection", {"selection_id": data["selection_id"]}, "c2")
        listed.update(data)
        return AIMessage(content="done")

    async def run():
        async with mcp_tools(tiny_tsv(tmp_path)) as tools:
            graph = build_graph(FnModel(messages=iter([]), reply=reply), tools)
            return await ask(graph, "list PEN")

    assert asyncio.run(run()) == "done"
    assert listed["total"] == 3  # the PEN ticket resolved in the same server session
    assert len(listed["neurons"]) == 3
