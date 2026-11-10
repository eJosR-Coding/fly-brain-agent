"""A turn that takes too long (e.g. a provider retrying 503s with backoff) must end with a
clear error event instead of leaving the client staring at keep-alive pings."""

import asyncio

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from flybrain.agent import build_graph, stream_events
from fakes import call, fn_model


@tool
async def slow_lookup(q: str) -> str:
    """Pretend to be a tool (or a provider) that hangs."""
    await asyncio.sleep(5)
    return "too late"


def _collect(graph, **kw):
    async def go():
        return [ev async for ev in stream_events(graph, "hi", "t1", **kw)]

    return asyncio.run(go())


def _model():
    def reply(messages):
        if any(getattr(m, "type", "") == "tool" for m in messages):
            return AIMessage(content="done")
        return call("slow_lookup", {"q": "x"})

    return fn_model(reply)


def test_slow_turn_ends_with_a_timeout_error():
    graph = build_graph(_model(), [slow_lookup])
    events = _collect(graph, turn_timeout=0.3)
    assert events[-1]["type"] == "error"
    assert "took longer than 0.3 s" in events[-1]["message"]


def test_no_timeout_by_default_lets_fast_turns_finish():
    @tool
    async def fast_lookup(q: str) -> str:
        """Instant tool."""
        return "ok"

    def reply(messages):
        if any(getattr(m, "type", "") == "tool" for m in messages):
            return AIMessage(content="done")
        return call("fast_lookup", {"q": "x"})

    graph = build_graph(fn_model(reply), [fast_lookup])
    events = _collect(graph, turn_timeout=5)
    assert events[-1] == {"type": "answer", "text": "done"}
