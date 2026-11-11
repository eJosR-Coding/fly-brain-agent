"""compact_history: what is SENT to the model is trimmed, what is CHECKPOINTED stays whole."""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from flybrain.agent import (
    MAX_TOOL_CHARS,
    build_graph,
    compact_history,
    find_dangling_tool_calls,
    keep_turns_from_env,
)
from fakes import call, fn_model


def _turn(n, rounds=1, answer=True, size=10):
    """One user turn: human, `rounds` x (AI tool_call + ToolMessage), then a final answer."""
    msgs = [HumanMessage(f"q{n}")]
    for r in range(rounds):
        cid = f"t{n}_{r}"
        msgs.append(AIMessage(content=f"thinking {n}.{r}", tool_calls=[{"name": "x", "args": {}, "id": cid}]))
        msgs.append(ToolMessage(content="r" * size, tool_call_id=cid, name="x"))
    if answer:
        msgs.append(AIMessage(content=f"a{n}"))
    return msgs


def _conv(turns, **kw):
    return [m for n in range(turns) for m in _turn(n, **kw)]


def _texts(msgs):
    return [m.text for m in msgs if isinstance(m, (HumanMessage, AIMessage)) and not getattr(m, "tool_calls", None)]


def test_short_conversation_is_untouched():
    msgs = _conv(3, rounds=2)
    assert compact_history(msgs, keep_turns=3) == msgs


def test_old_turns_keep_only_question_and_final_answer():
    msgs = _conv(5, rounds=2)
    out = compact_history(msgs, keep_turns=3)
    old = out[:4]  # turns 0 and 1 -> human + final answer each
    assert [type(m).__name__ for m in old] == ["HumanMessage", "AIMessage"] * 2
    assert [m.text for m in old] == ["q0", "a0", "q1", "a1"]
    assert out[4:] == msgs[len(_conv(2, rounds=2)):]  # last 3 turns whole


def test_counts_kept_and_dropped():
    msgs = _conv(5, rounds=2)  # 5 turns x (1 + 4 + 1) = 30 messages
    out = compact_history(msgs, keep_turns=3)
    assert len(msgs) == 30 and len(out) == 3 * 6 + 2 * 2
    assert len(msgs) - len(out) == 2 * 4  # 4 tool-related messages dropped per old turn


def test_pairing_invariant_holds_after_compaction():
    msgs = _conv(6, rounds=3)
    for keep in (1, 2, 3, 5, 6, 9):
        assert find_dangling_tool_calls(compact_history(msgs, keep_turns=keep)) == [], keep
    # no orphan ToolMessage either
    out = compact_history(msgs, keep_turns=2)
    asked = {tc["id"] for m in out if isinstance(m, AIMessage) for tc in m.tool_calls}
    assert all(m.tool_call_id in asked for m in out if isinstance(m, ToolMessage))


def test_idempotent():
    msgs = _conv(6, rounds=2, size=9000)
    once = compact_history(msgs, keep_turns=2)
    assert compact_history(once, keep_turns=2) == once


def test_turn_in_progress_is_kept_whole():
    msgs = _conv(4, rounds=1) + [HumanMessage("now")] + _turn(9, rounds=0)[1:0]
    msgs += [AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "cur"}])]
    out = compact_history(msgs, keep_turns=1)
    assert out[-2:] == msgs[-2:]  # human + its pending tool call, no answer yet
    # an unanswered call in the current turn is not "fixed" or dropped by compaction
    assert [c["id"] for c in find_dangling_tool_calls(out)] == ["cur"]


def test_current_turn_with_tool_results_survives_keep_one():
    msgs = _conv(3, rounds=1) + [HumanMessage("now")] + _turn(7, rounds=2, answer=False)[1:]
    out = compact_history(msgs, keep_turns=1)
    cur = msgs[msgs.index(next(m for m in msgs if m.text == "now")):]
    assert out[-len(cur):] == cur


def test_old_turn_without_a_final_answer_keeps_only_the_question():
    msgs = _turn(0, rounds=2, answer=False) + _conv(1)[0:0] + _turn(1) + _turn(2)
    out = compact_history(msgs, keep_turns=1)
    assert [m.text for m in out[:2]] == ["q0", "q1"]
    assert find_dangling_tool_calls(out) == []


def test_oversized_tool_results_in_the_window_are_truncated_with_a_marker():
    big = ToolMessage(content="z" * 10_000, tool_call_id="b", name="x")
    msgs = [HumanMessage("q"), AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "b"}]), big]
    out = compact_history(msgs, keep_turns=3)
    cut = out[2]
    assert isinstance(cut, ToolMessage) and cut.tool_call_id == "b" and cut.name == "x"
    assert len(cut.content) <= MAX_TOOL_CHARS
    assert "[truncated" in cut.content and cut.content.startswith("zzz")
    assert big.content == "z" * 10_000  # the input is never mutated (it is the checkpoint)


def test_small_tool_results_are_not_touched():
    msgs = _conv(2, size=MAX_TOOL_CHARS)
    assert compact_history(msgs, keep_turns=3) == msgs


def test_input_list_is_not_mutated():
    msgs = _conv(5, rounds=2, size=9000)
    snapshot = [m.model_copy() for m in msgs]
    compact_history(msgs, keep_turns=1)
    assert msgs == snapshot


def test_keep_turns_must_be_at_least_one():
    with pytest.raises(ValueError, match="keep_turns"):
        compact_history(_conv(2), keep_turns=0)


# --- configuration ---------------------------------------------------------------------------

def test_keep_turns_default_and_env(monkeypatch):
    monkeypatch.delenv("FLYBRAIN_KEEP_TURNS", raising=False)
    assert keep_turns_from_env() == 3
    monkeypatch.setenv("FLYBRAIN_KEEP_TURNS", "5")
    assert keep_turns_from_env() == 5


@pytest.mark.parametrize("bad", ["0", "-1", "two", "1.5", " "])
def test_keep_turns_env_is_validated(monkeypatch, bad):
    monkeypatch.setenv("FLYBRAIN_KEEP_TURNS", bad)
    with pytest.raises(ValueError, match="FLYBRAIN_KEEP_TURNS"):
        keep_turns_from_env()


# --- end to end: the model sees the compact list, the checkpoint keeps everything -------------

@tool
def lookup(q: str) -> str:
    """Return a bulky payload."""
    return json.dumps({"q": q, "blob": "x" * 3000})


def _scripted_model():
    def reply(msgs):
        if msgs[-1].type == "human":
            return call("lookup", {"q": msgs[-1].text}, cid=f"c{len(msgs)}")
        return AIMessage(content=f"answer to {[m for m in msgs if m.type == 'human'][-1].text}")
    return fn_model(reply)


def _sent_chars(model, call_index):
    return sum(len(m.text) for m in model.seen[call_index])


async def _run(keep_turns, questions):
    model = _scripted_model()
    graph = build_graph(model, [lookup], checkpointer=InMemorySaver(), keep_turns=keep_turns)
    cfg = {"configurable": {"thread_id": "t"}}
    from langchain_core.messages import HumanMessage as H
    for q in questions:
        await graph.ainvoke({"messages": [H(q)]}, config=cfg)
    state = await graph.aget_state(cfg)
    return model, state.values["messages"]


def test_model_input_is_compacted_but_the_checkpoint_is_whole():
    qs = [f"question {i}" for i in range(5)]
    model, stored = asyncio.run(_run(2, qs))
    assert sum(1 for m in stored if isinstance(m, ToolMessage)) == 5  # full memory stored
    last_input = model.seen[-1]
    assert sum(1 for m in last_input if isinstance(m, ToolMessage)) == 2  # only the 2 newest turns
    assert find_dangling_tool_calls(last_input) == []
    # older questions are still there, as text
    assert {"question 0", "question 1", "answer to question 0"} <= {m.text for m in last_input}


def test_compaction_shrinks_what_is_sent():
    qs = [f"question {i}" for i in range(6)]
    big, _ = asyncio.run(_run(99, qs))
    small, _ = asyncio.run(_run(2, qs))
    assert _sent_chars(small, -1) < _sent_chars(big, -1) / 2
