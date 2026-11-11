"""stream_events: the agent loop as typed dicts (tool_call, tool_result, answer, error)."""

import asyncio
import json

import anyio

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
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
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "usage", "answer"]
    assert events[0] == {"type": "tool_call", "id": "c1", "tool": "count_neurons", "args": {"cell_type": "PEN"}}
    assert (events[1]["id"], events[1]["tool"], events[1]["error"]) == ("c1", "count_neurons", False)
    assert isinstance(events[1]["ms"], int) and events[1]["ms"] >= 0
    assert json.loads(events[1]["content"])["count"] == 42
    assert events[3]["text"] == "42 PEN"


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
    assert events[-1] == {"type": "answer", "text": "[no answer] the model returned an empty reply"}
    assert [e["type"] for e in events] == ["usage", "answer"]


class StatusError(Exception):
    def __init__(self, text, status_code=None, code=None):
        super().__init__(text)
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code


def failing_model(exc):
    def reply(msgs):
        raise exc

    return fn_model(reply)


def error_message(exc, model_name="groq:fast"):
    graph = build_graph(failing_model(exc), [count_neurons], checkpointer=InMemorySaver())
    events = asyncio.run(collect(graph, "q", model_name=model_name))
    assert [e["type"] for e in events] == ["error"]
    return events[0]["message"]


def test_quota_errors_are_friendly():
    for exc in (
        StatusError("Error code: 429 - rate_limit_exceeded", status_code=429),
        StatusError("429 RESOURCE_EXHAUSTED. You exceeded your current quota, limit: 20", code=429),
        RuntimeError("Rate limit reached for model on tokens per day"),
    ):
        msg = error_message(exc)
        assert msg.startswith("model groq:fast hit its quota or rate limit; switch model in the picker or retry later")
        assert len(msg) < 300


def test_auth_errors_are_friendly():
    for exc in (
        StatusError("Error code: 401 - invalid_api_key", status_code=401),
        StatusError("403 PERMISSION_DENIED", code=403),
        RuntimeError("API key not valid. Please pass a valid API key."),
    ):
        assert error_message(exc) == "model groq:fast: API key missing or invalid"


def test_other_provider_errors_keep_the_generic_text():
    assert error_message(RuntimeError("boom")) == "model call failed: boom"
    assert error_message(StatusError("server error", status_code=500)) == "model call failed: server error"


def test_friendly_text_without_a_model_name_says_the_model():
    msg = error_message(StatusError("429", status_code=429), model_name=None)
    assert msg.startswith("the model hit its quota or rate limit")


class StreamingModel(GenericFakeChatModel):
    """Streams its reply in word chunks; tool calls are non-streamed."""

    reply: object = None
    disable_streaming: bool = False

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.reply(messages))])

    def _stream(self, messages, *args, **kwargs):
        msg = self.reply(messages)
        if msg.tool_calls:
            yield ChatGenerationChunk(message=AIMessageChunk(
                content=msg.content, tool_call_chunks=[
                    {"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": i}
                    for i, tc in enumerate(msg.tool_calls)]))
            return
        for word in msg.content.split(" "):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))


def streaming(reply) -> StreamingModel:
    return StreamingModel(messages=iter([]), reply=reply)


def test_final_answer_streams_as_deltas_then_the_full_answer():
    graph = build_graph(streaming(lambda m: AIMessage(content="one two three")), [count_neurons],
                        checkpointer=InMemorySaver())
    events = asyncio.run(collect(graph, "q"))
    types = [e["type"] for e in events]
    assert types[-1] == "answer" and set(types[:-2]) == {"answer_delta"} and types[-2] == "usage"
    assert len(types) > 2
    assert "".join(e["text"] for e in events if e["type"] == "answer_delta").strip() == "one two three"
    assert events[-1]["text"].strip() == "one two three"


def test_tool_call_arguments_are_never_streamed_as_answer_text():
    def reply(msgs):
        if msgs[-1].type == "human":
            return call("count_neurons", {"cell_type": "PEN"})
        return AIMessage(content="42 PEN")

    graph = build_graph(streaming(reply), [count_neurons], checkpointer=InMemorySaver())
    events = asyncio.run(collect(graph, "q"))
    deltas = "".join(e["text"] for e in events if e["type"] == "answer_delta")
    assert "cell_type" not in deltas and deltas.strip() == "42 PEN"
    assert [e["type"] for e in events if e["type"] != "answer_delta"] == [
        "tool_call", "tool_result", "usage", "answer"]


def test_non_streaming_models_just_send_the_answer():
    graph = build_graph(fn_model(lambda m: AIMessage(content="plain")), [count_neurons],
                        checkpointer=InMemorySaver())
    assert [e["type"] for e in asyncio.run(collect(graph, "q"))] == ["usage", "answer"]
