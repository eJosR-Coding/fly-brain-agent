"""stream_events: the agent loop as typed dicts (tool_call, tool_result, answer, error)."""

import asyncio
import json

import anyio

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from flybrain.agent import build_graph, stream_events
from fakes import call, fn_model


@tool
def count_neurons(cell_type: str) -> str:
    """Count neurons of a cell type."""
    return json.dumps({"cell_type": cell_type, "count": 42})


@tool
def dead_server(x: str) -> str:
    """Pretend the MCP server died."""
    raise anyio.ClosedResourceError()


async def collect(graph, question, thread_id="t1", **kw):
    return [e async for e in stream_events(graph, question, thread_id, **kw)]


def test_tool_call_result_answer_in_order():
    def reply(msgs):
        if msgs[-1].type == "human":
            return call("count_neurons", {"cell_type": "PEN"})
        return AIMessage(content="42 PEN")

    graph = build_graph(fn_model(reply), [count_neurons], checkpointer=InMemorySaver())
    events = asyncio.run(collect(graph, "how many?"))
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "answer"]
    assert events[0] == {"type": "tool_call", "tool": "count_neurons", "args": {"cell_type": "PEN"}}
    assert json.loads(events[1]["content"])["count"] == 42
    assert events[2]["text"] == "42 PEN"


def test_same_thread_remembers_previous_turn():
    model = fn_model(lambda msgs: AIMessage(content="ok"))
    graph = build_graph(model, [count_neurons], checkpointer=InMemorySaver())

    async def run():
        await collect(graph, "first", "t1")
        await collect(graph, "second", "t1")
        await collect(graph, "other", "t2")

    asyncio.run(run())
    humans = [sum(isinstance(m, HumanMessage) for m in seen) for seen in model.seen]
    assert humans == [1, 2, 1]


def test_tool_budget_is_per_turn_not_per_conversation():
    n = {"i": 0}

    def reply(msgs):
        if msgs[-1].type == "human":
            n["i"] += 1
            return call("count_neurons", {"cell_type": "PEN"}, f"c{n['i']}")
        return AIMessage(content="done")

    graph = build_graph(fn_model(reply), [count_neurons], checkpointer=InMemorySaver())

    async def run():
        out = []
        for _ in range(3):  # budget is 1 tool round; every turn uses exactly 1
            out.append(await collect(graph, "q", "t1", max_steps=1))
        return out

    for events in asyncio.run(run()):
        assert events[-1] == {"type": "answer", "text": "done"}


def test_looping_model_ends_with_hard_stop_error():
    i = {"n": 0}

    def reply(msgs):
        i["n"] += 1
        return call("count_neurons", {"cell_type": "PEN"}, f"c{i['n']}")

    graph = build_graph(fn_model(reply), [count_neurons], checkpointer=InMemorySaver())
    events = asyncio.run(collect(graph, "loop", max_steps=2))
    assert events[-1]["type"] == "error"
    assert "hard stop" in events[-1]["message"]
    assert not any(e["type"] == "answer" for e in events)


def test_model_failure_becomes_error_event():
    def reply(msgs):
        raise RuntimeError("provider exploded")

    graph = build_graph(fn_model(reply), [count_neurons], checkpointer=InMemorySaver())
    events = asyncio.run(collect(graph, "hi"))
    assert events == [{"type": "error", "message": "model call failed: provider exploded"}]


def test_server_stopped_becomes_error_event():
    graph = build_graph(
        fn_model(lambda m: call("dead_server", {"x": "a"}) if m[-1].type == "human" else AIMessage(content="x")),
        [dead_server],
        checkpointer=InMemorySaver(),
    )
    events = asyncio.run(collect(graph, "hi"))
    assert events[-1] == {"type": "error", "message": "the flybrain server stopped: ClosedResourceError"}


def test_empty_reply_is_reported():
    graph = build_graph(fn_model(lambda m: AIMessage(content="")), [count_neurons], checkpointer=InMemorySaver())
    events = asyncio.run(collect(graph, "hi"))
    assert events == [{"type": "answer", "text": "[no answer] the model returned an empty reply"}]
