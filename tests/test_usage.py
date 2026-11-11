"""Token usage per turn: summed from the usage_metadata of every model call, null when unknown."""

import asyncio
import json

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from flybrain.agent import build_graph, stream_events, sum_usage
from flybrain.api import create_app
from fakes import call, fn_model
from test_api_chat import chat, parse_sse


@tool
def count_neurons(cell_type: str) -> str:
    """Count neurons of a cell type."""
    return json.dumps({"cell_type": cell_type, "count": 42})


def usage(i, o, total=None):
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o if total is None else total}


def two_calls(first, second):
    def reply(msgs):
        if msgs[-1].type == "human":
            msg = call("count_neurons", {"cell_type": "PEN"})
            msg.usage_metadata = first
            return msg
        return AIMessage(content="42 PEN", usage_metadata=second)
    return reply


def run(model, **kw):
    graph = build_graph(model, [count_neurons], checkpointer=InMemorySaver())

    async def go():
        return [e async for e in stream_events(graph, "q", "t1", **kw)]
    return asyncio.run(go())


def usage_event(events):
    found = [e for e in events if e["type"] == "usage"]
    assert len(found) == 1
    return found[0]


# --- the pure sum ---------------------------------------------------------------------------

def test_sum_usage_adds_known_fields_and_keeps_unknown_as_none():
    assert sum_usage([usage(10, 2), usage(20, 3)]) == {"input_tokens": 30, "output_tokens": 5, "total_tokens": 35}
    assert sum_usage([None, None]) == {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    assert sum_usage([]) == {"input_tokens": None, "output_tokens": None, "total_tokens": None}


def test_sum_usage_derives_total_when_the_provider_omits_it():
    only_io = {"input_tokens": 7, "output_tokens": 3}
    assert sum_usage([only_io])["total_tokens"] == 10


def test_sum_usage_skips_calls_without_usage_but_sums_the_rest():
    assert sum_usage([usage(10, 2), None])["total_tokens"] == 12


# --- stream_events ---------------------------------------------------------------------------

def test_usage_event_sums_every_model_call_and_comes_before_the_answer():
    events = run(fn_model(two_calls(usage(100, 10), usage(150, 20))))
    ev = usage_event(events)
    assert ev == {"type": "usage", "calls": 2, "input_tokens": 250, "output_tokens": 30, "total_tokens": 280}
    types = [e["type"] for e in events]
    assert types.index("usage") < types.index("answer") and types[-1] == "answer"


def test_missing_usage_is_null_not_zero():
    ev = usage_event(run(fn_model(two_calls(None, None))))
    assert ev["calls"] == 2
    assert ev["input_tokens"] is None and ev["output_tokens"] is None and ev["total_tokens"] is None


def test_partial_usage_sums_what_is_known():
    ev = usage_event(run(fn_model(two_calls(None, usage(5, 1)))))
    assert (ev["calls"], ev["input_tokens"], ev["total_tokens"]) == (2, 5, 6)


class StreamingUsageModel(GenericFakeChatModel):
    """Streams word chunks; token usage rides only on the FINAL chunk, like real providers."""

    reply: object = None
    seen: list = []
    disable_streaming: bool = False

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.reply(messages))])

    def _stream(self, messages, *args, **kwargs):
        msg = self.reply(messages)
        if msg.tool_calls:
            yield ChatGenerationChunk(message=AIMessageChunk(
                content="", tool_call_chunks=[
                    {"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": 0}
                    for tc in msg.tool_calls]))
        else:
            for word in msg.content.split(" "):
                yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))
        yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=msg.usage_metadata))


def test_streamed_usage_on_the_final_chunk_is_counted_once_per_call():
    model = StreamingUsageModel(messages=iter([]), reply=two_calls(usage(100, 10), usage(150, 20)), seen=[])
    events = run(model)
    assert any(e["type"] == "answer_delta" for e in events)  # it really streamed
    assert usage_event(events) == {
        "type": "usage", "calls": 2, "input_tokens": 250, "output_tokens": 30, "total_tokens": 280,
    }


def test_a_failed_turn_still_reports_the_usage_it_spent():
    def reply(msgs):
        msg = call("count_neurons", {"cell_type": "PEN"}, cid=f"c{len(msgs)}")
        msg.usage_metadata = usage(10, 1)
        return msg

    events = run(fn_model(reply), max_steps=1)
    assert events[-1]["type"] == "error"
    ev = usage_event(events)
    assert ev["calls"] == 2 and ev["total_tokens"] == 22
    assert [e["type"] for e in events].index("usage") < len(events) - 1


def test_no_model_call_means_no_usage_event():
    def boom(msgs):
        raise RuntimeError("provider down")

    events = run(fn_model(boom))
    assert events[-1]["type"] == "error"
    assert [e for e in events if e["type"] == "usage"] == []


# --- over SSE ------------------------------------------------------------------------------------

def test_chat_sse_carries_usage_before_done(tiny_tsv_path):
    model = fn_model(two_calls(usage(100, 10), usage(150, 20)))
    with TestClient(create_app(atlas_path=tiny_tsv_path, model=model)) as client:
        events = parse_sse(chat(client, "hi").text)
    names = [n for n, _ in events]
    assert names[-1] == "done" and names.index("usage") < names.index("done")
    assert dict(events)["usage"] == {"calls": 2, "input_tokens": 250, "output_tokens": 30, "total_tokens": 280}


def test_chat_sse_usage_is_null_when_the_provider_reports_none(tiny_tsv_path):
    model = fn_model(lambda msgs: AIMessage(content="hello"))
    with TestClient(create_app(atlas_path=tiny_tsv_path, model=model)) as client:
        events = parse_sse(chat(client, "hi").text)
    assert dict(events)["usage"] == {"calls": 1, "input_tokens": None, "output_tokens": None, "total_tokens": None}
