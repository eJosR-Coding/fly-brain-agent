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


# --- hardening (H3) ---------------------------------------------------------------

import anyio
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from flybrain.agent_mcp import ServerStoppedError

BOOM: list = []  # what `flaky` raises next (set per test)


@tool
def flaky(cell_type: str) -> str:
    """A tool whose backend can die."""
    raise BOOM[0]


def invalid_call(call_id="c1"):
    return AIMessage(
        content="",
        invalid_tool_calls=[
            {"name": "count_neurons", "args": '{"cell_type": ', "id": call_id,
             "error": "could not parse JSON", "type": "invalid_tool_call"}
        ],
    )


@pytest.mark.parametrize("bad", [0, -1, 1.5, "3", True, None])
def test_max_steps_must_be_a_positive_int(bad):
    async def run():
        graph = build_graph(make_model([AIMessage(content="hi")]), [count_neurons])
        with pytest.raises(ValueError, match="max_steps must be an int >= 1"):
            await ask(graph, "q", max_steps=bad)

    asyncio.run(run())


def test_one_tool_round_plus_the_answer_fits_in_max_steps_1():
    async def run():
        model = make_model([tool_call("count_neurons", {"cell_type": "PEN"}), AIMessage(content="42")])
        assert await ask(build_graph(model, [count_neurons]), "q", max_steps=1) == "42"

    asyncio.run(run())


def test_two_tool_rounds_need_max_steps_2():
    async def run():
        def script():
            return [tool_call("count_neurons", {"cell_type": "A"}, "c1"),
                    tool_call("count_neurons", {"cell_type": "B"}, "c2"),
                    AIMessage(content="done")]

        graph = build_graph(make_model(script()), [count_neurons])
        assert "hard stop" in await ask(graph, "q", max_steps=1)
        graph = build_graph(make_model(script()), [count_neurons])
        assert await ask(graph, "q", max_steps=2) == "done"

    asyncio.run(run())


def test_hard_stop_message_counts_tool_rounds():
    async def run():
        forever = (tool_call("count_neurons", {"cell_type": "PEN"}, f"c{i}") for i in range(1000))
        graph = build_graph(FakeModel(messages=forever, seen=[]), [count_neurons])
        assert await ask(graph, "loop", max_steps=3) == (
            "[hard stop] reached 3 tool rounds without a final answer"
        )
        assert len(CALLS) == 3

    asyncio.run(run())


def test_malformed_tool_call_is_fed_back_so_the_model_can_retry():
    async def run():
        model = make_model([invalid_call("c9"), AIMessage(content="fixed it")])
        answer = await ask(build_graph(model, [count_neurons]), "q")
        assert answer == "fixed it"
        retry_view = model.seen[1]
        fb = [m for m in retry_view if isinstance(m, ToolMessage)]
        assert len(fb) == 1 and fb[0].tool_call_id == "c9" and fb[0].status == "error"
        assert "could not parse JSON" in fb[0].content

    asyncio.run(run())


def test_malformed_call_without_id_still_gets_feedback():
    async def run():
        bad = AIMessage(content="", invalid_tool_calls=[
            {"name": None, "args": None, "id": None, "error": "nope", "type": "invalid_tool_call"}])
        model = make_model([bad, AIMessage(content="ok")])
        assert await ask(build_graph(model, [count_neurons]), "q") == "ok"

    asyncio.run(run())


@pytest.mark.parametrize("content", ["", "   \n"])
def test_empty_final_answer_is_reported_not_swallowed(content):
    async def run():
        graph = build_graph(make_model([AIMessage(content=content)]), [count_neurons])
        assert await ask(graph, "q") == "[no answer] the model returned an empty reply"

    asyncio.run(run())


DEAD = [
    lambda: anyio.ClosedResourceError(),
    lambda: anyio.BrokenResourceError(),
    lambda: McpError(ErrorData(code=-32000, message="Connection closed")),
    lambda: ExceptionGroup("tg", [anyio.ClosedResourceError()]),
    lambda: BrokenPipeError("pipe"),
]


@pytest.mark.parametrize("make", DEAD)
def test_dead_server_stops_the_run_with_a_clear_error(make):
    async def run():
        BOOM[:] = [make()]
        model = make_model([tool_call("flaky", {"cell_type": "x"}), AIMessage(content="made it up")])
        graph = build_graph(model, [flaky])
        with pytest.raises(ServerStoppedError, match="the flybrain server stopped: ") as exc:
            await ask(graph, "q")
        assert "made it up" not in str(exc.value)
        assert len(model.seen) == 1  # the model never got a second turn

    asyncio.run(run())


def test_dead_server_message_carries_the_cause():
    async def run():
        BOOM[:] = [McpError(ErrorData(code=-32000, message="Connection closed"))]
        graph = build_graph(make_model([tool_call("flaky", {"cell_type": "x"})]), [flaky])
        with pytest.raises(ServerStoppedError, match="stopped: Connection closed"):
            await ask(graph, "q")

    asyncio.run(run())


def test_ordinary_tool_errors_still_go_back_to_the_model():
    async def run():
        BOOM[:] = [ValueError("bad cell_type")]
        model = make_model([tool_call("flaky", {"cell_type": "x"}), AIMessage(content="recovered")])
        assert await ask(build_graph(model, [flaky]), "q") == "recovered"
        fb = [m for m in model.seen[1] if isinstance(m, ToolMessage)]
        assert fb[0].status == "error" and "bad cell_type" in fb[0].content

    asyncio.run(run())
