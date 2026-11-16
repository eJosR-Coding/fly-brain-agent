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


def test_provider_timeout_without_a_deadline_is_an_ordinary_failure():
    def reply(messages):
        raise TimeoutError("read timed out")  # the provider SDK's own timeout, not our deadline

    graph = build_graph(fn_model(reply), [slow_lookup])
    events = _collect(graph)  # turn_timeout=None
    assert events == [{"type": "error", "message": "model call failed: read timed out"}]


def test_provider_timeout_before_our_deadline_is_an_ordinary_failure():
    def reply(messages):
        raise TimeoutError("read timed out")

    graph = build_graph(fn_model(reply), [slow_lookup])
    events = _collect(graph, turn_timeout=60)
    assert events[-1] == {"type": "error", "message": "model call failed: read timed out"}


def test_a_listed_tool_extends_the_deadline():
    @tool
    async def experiment(q: str) -> str:
        """A long experiment tool (0.6 s here)."""
        await asyncio.sleep(0.6)
        return "ok"

    def reply(messages):
        if any(getattr(m, "type", "") == "tool" for m in messages):
            return AIMessage(content="done")
        return call("experiment", {"q": "x"})

    graph = build_graph(fn_model(reply), [experiment])
    cut = _collect(graph, turn_timeout=0.3)
    assert cut[-1]["type"] == "error" and "took longer than 0.3 s" in cut[-1]["message"]
    graph = build_graph(fn_model(reply), [experiment])
    events = _collect(graph, turn_timeout=0.3, tool_extensions={"experiment": 5.0})
    assert events[-1] == {"type": "answer", "text": "done"}


def test_the_extension_does_not_apply_to_other_tools():
    graph = build_graph(_model(), [slow_lookup])
    events = _collect(graph, turn_timeout=0.3, tool_extensions={"experiment": 60.0})
    assert events[-1]["type"] == "error" and "took longer than 0.3 s" in events[-1]["message"]


# --- the experiment grant is held only while the call is in flight (S14) ---------------------

def _tools():
    @tool
    async def experiment(q: str) -> str:
        """A long experiment tool (0.6 s here)."""
        await asyncio.sleep(0.6)
        return "ok"

    @tool
    async def failing(q: str) -> str:
        """Fails at once."""
        raise ValueError("bad args")

    @tool
    async def quick(q: str) -> str:
        """Instant."""
        return "ok"

    return experiment, failing, quick


def _calls(*names):
    return AIMessage(
        content="",
        tool_calls=[{"name": n, "args": {"q": "x"}, "id": f"id_{n}"} for n in names],
    )


def test_parallel_experiment_and_quick_tool_keep_the_grant_until_both_finish():
    experiment, _, quick = _tools()

    def reply(messages):
        if any(getattr(m, "type", "") == "tool" for m in messages):
            return AIMessage(content="done")
        return _calls("experiment", "quick")

    graph = build_graph(fn_model(reply), [experiment, quick])
    events = _collect(graph, turn_timeout=0.3, tool_extensions={"experiment": 5.0})
    assert events[-1] == {"type": "answer", "text": "done"}


def test_the_grant_is_released_when_the_call_ends_early():
    import time

    _, failing, _ = _tools()

    def reply(messages):
        tools_seen = sum(getattr(m, "type", "") == "tool" for m in messages)
        if tools_seen == 0:
            return _calls("failing")
        time.sleep(0.8)  # a stuck provider AFTER the failed experiment must not enjoy its grant
        return AIMessage(content="too late")

    graph = build_graph(fn_model(reply), [failing])
    events = _collect(graph, turn_timeout=0.4, tool_extensions={"failing": 30.0})
    assert events[-1]["type"] == "error" and "took longer than 0.4 s" in events[-1]["message"]


def test_a_call_cut_by_the_hard_stop_ends_with_the_hard_stop_not_a_timeout():
    experiment, _, _ = _tools()
    graph = build_graph(fn_model(lambda m: _calls("experiment")), [experiment])
    events = _collect(graph, turn_timeout=5, max_steps=1, tool_extensions={"experiment": 600.0})
    assert events[-1]["type"] == "error" and "hard stop" in events[-1]["message"]
