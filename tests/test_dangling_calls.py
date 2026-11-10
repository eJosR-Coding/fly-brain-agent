"""A cut turn must not leave an unpaired tool_call in the checkpointed thread: providers reject
such a history, so the conversation would be poisoned for good."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from flybrain.agent import (
    build_graph,
    close_dangling_tool_calls,
    find_dangling_tool_calls,
    stream_events,
)
from fakes import call, fn_model


@tool
async def quick(q: str) -> str:
    """Instant tool."""
    return "ok"


@tool
async def slow(q: str) -> str:
    """Hangs."""
    await asyncio.sleep(5)
    return "too late"


def _ai(*ids):
    return AIMessage(content="", tool_calls=[{"name": "quick", "args": {}, "id": i} for i in ids])


def _tm(cid):
    return ToolMessage(content="ok", tool_call_id=cid, name="quick")


def assert_paired(messages):
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    asked = [tc["id"] for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls]
    assert set(asked) <= answered, f"dangling: {set(asked) - answered}"


# --- pure helper -------------------------------------------------------------------------

def test_find_none_dangling():
    assert find_dangling_tool_calls([HumanMessage("hi"), _ai("a"), _tm("a"), AIMessage("done")]) == []


def test_find_one_dangling():
    found = find_dangling_tool_calls([HumanMessage("hi"), _ai("a")])
    assert [(c["id"], c["name"]) for c in found] == [("a", "quick")]


def test_find_several_dangling():
    found = find_dangling_tool_calls([HumanMessage("hi"), _ai("a", "b"), _ai("c")])
    assert [c["id"] for c in found] == ["a", "b", "c"]


def test_find_partial_pairing():
    found = find_dangling_tool_calls([HumanMessage("hi"), _ai("a", "b"), _tm("a")])
    assert [c["id"] for c in found] == ["b"]


# --- end to end ---------------------------------------------------------------------------

async def collect(graph, q, thread="t1", **kw):
    return [e async for e in stream_events(graph, q, thread, **kw)]


def _model(tool_name):
    n = {"i": 0}

    def reply(messages):
        if messages[-1].type == "human" and "second" in messages[-1].text:
            return AIMessage(content="recovered")
        n["i"] += 1
        return call(tool_name, {"q": "x"}, cid=f"call{n['i']}")

    return fn_model(reply)


def test_hard_stop_leaves_thread_usable():
    model = _model("quick")
    graph = build_graph(model, [quick], checkpointer=InMemorySaver())

    async def run():
        first = await collect(graph, "first", max_steps=1)
        second = await collect(graph, "second", max_steps=1)
        return first, second

    first, second = asyncio.run(run())
    assert "hard stop" in first[-1]["message"]
    assert second[-1] == {"type": "answer", "text": "recovered"}
    assert_paired(model.seen[-1])
    cancelled = [m for m in model.seen[-1] if isinstance(m, ToolMessage) and m.status == "error"]
    assert len(cancelled) == 1 and cancelled[0].content.startswith("cancelled:")


def test_timeout_mid_tool_leaves_thread_usable():
    model = _model("slow")
    graph = build_graph(model, [slow], checkpointer=InMemorySaver())

    async def run():
        first = await collect(graph, "first", turn_timeout=0.3)
        second = await collect(graph, "second", turn_timeout=5)
        return first, second

    first, second = asyncio.run(run())
    assert "took longer" in first[-1]["message"]
    assert second[-1] == {"type": "answer", "text": "recovered"}
    assert_paired(model.seen[-1])


def test_normal_turn_adds_no_synthetic_messages():
    def reply(messages):
        if messages[-1].type == "tool":
            return AIMessage(content="done")
        return call("quick", {"q": "x"})

    graph = build_graph(fn_model(reply), [quick], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    async def run():
        events = await collect(graph, "hi")
        return events, (await graph.aget_state(config)).values["messages"]

    events, messages = asyncio.run(run())
    assert events[-1]["type"] == "answer"
    assert [m.type for m in messages] == ["human", "ai", "tool", "ai"]
    assert not any(isinstance(m, ToolMessage) and m.status == "error" for m in messages)


def test_close_without_checkpointer_is_a_noop():
    graph = build_graph(fn_model(lambda m: AIMessage(content="x")), [quick])
    asyncio.run(close_dangling_tool_calls(graph, {"configurable": {"thread_id": "t"}}, "why"))
