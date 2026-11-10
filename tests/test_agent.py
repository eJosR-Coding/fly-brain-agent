"""Agent loop tests: scripted fake model, fake tools, no network, no API key."""

import asyncio
import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from flybrain.agent import MAX_STEPS, SYSTEM_PROMPT, ask, build_graph

CALLS: list[dict] = []


@tool
def count_neurons(cell_type: str) -> str:
    """Count neurons of a cell type."""
    CALLS.append({"cell_type": cell_type})
    return json.dumps({"cell_type": cell_type, "count": 42})


class FakeModel(GenericFakeChatModel):
    """Scripted model that tolerates bind_tools and records what it was shown."""

    seen: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def tool_call(name, args, call_id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def make_model(script):
    return FakeModel(messages=iter(script), seen=[])


@pytest.fixture(autouse=True)
def _reset_calls():
    CALLS.clear()


def test_tool_then_answer():
    async def run():
        model = make_model([tool_call("count_neurons", {"cell_type": "PEN"}), AIMessage(content="42 PEN neurons")])
        graph = build_graph(model, [count_neurons])
        answer = await ask(graph, "how many PEN?")
        assert answer == "42 PEN neurons"
        assert CALLS == [{"cell_type": "PEN"}]

    asyncio.run(run())


def test_never_stops_hits_hard_stop():
    async def run():
        forever = (tool_call("count_neurons", {"cell_type": "PEN"}, f"c{i}") for i in range(1000))
        model = FakeModel(messages=forever, seen=[])
        graph = build_graph(model, [count_neurons])
        answer = await ask(graph, "loop", max_steps=5)
        assert "hard stop" in answer
        assert "5" in answer

    asyncio.run(run())


def test_unknown_tool_is_reported_to_model():
    async def run():
        model = make_model([tool_call("nope", {}), AIMessage(content="sorry")])
        graph = build_graph(model, [count_neurons])
        result = await graph.ainvoke({"messages": [HumanMessage("hi")]})
        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert "nope" in tool_msgs[0].content
        assert result["messages"][-1].content == "sorry"

    asyncio.run(run())


def test_system_prompt_comes_first():
    async def run():
        model = make_model([AIMessage(content="ok")])
        graph = build_graph(model, [count_neurons])
        await ask(graph, "hello")
        first = model.seen[0][0]
        assert first.type == "system"
        assert first.content == SYSTEM_PROMPT

    asyncio.run(run())


def test_max_steps_is_sane():
    assert MAX_STEPS >= 5
