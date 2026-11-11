"""fit_to_budget: every request sent to the model fits a token budget (chars / 4)."""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from flybrain.agent import (
    DEFAULT_CONTEXT_BUDGET,
    SYSTEM_PROMPT,
    TOOL_FLOOR_CHARS,
    build_graph,
    compact_history,
    context_budget_from_env,
    estimate_tokens,
    find_dangling_tool_calls,
    fit_to_budget,
    tool_definition_chars,
)
from fakes import call, fn_model


def _turn(n, sizes=(10,), answer=True):
    msgs = [HumanMessage(f"q{n}")]
    for r, size in enumerate(sizes):
        cid = f"t{n}_{r}"
        msgs.append(AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": cid}]))
        msgs.append(ToolMessage(content="r" * size, tool_call_id=cid, name="x"))
    if answer:
        msgs.append(AIMessage(content=f"a{n}"))
    return msgs


def _tokens(msgs, overhead=0):
    return estimate_tokens(msgs, overhead_chars=overhead)


def _tool_msgs(msgs):
    return [m for m in msgs if isinstance(m, ToolMessage)]


# --- configuration -------------------------------------------------------------------------

def test_budget_default_and_env(monkeypatch):
    monkeypatch.delenv("FLYBRAIN_CONTEXT_BUDGET", raising=False)
    assert context_budget_from_env() == DEFAULT_CONTEXT_BUDGET == 6000
    monkeypatch.setenv("FLYBRAIN_CONTEXT_BUDGET", "1000")
    assert context_budget_from_env() == 1000
    monkeypatch.setenv("FLYBRAIN_CONTEXT_BUDGET", "12000")
    assert context_budget_from_env() == 12000


@pytest.mark.parametrize("bad", ["999", "0", "-5", "lots", "6000.5", " "])
def test_budget_env_is_validated(monkeypatch, bad):
    monkeypatch.setenv("FLYBRAIN_CONTEXT_BUDGET", bad)
    with pytest.raises(ValueError, match="FLYBRAIN_CONTEXT_BUDGET"):
        context_budget_from_env()


def test_estimate_is_chars_over_four_including_tool_call_args_and_overhead():
    msgs = [HumanMessage("a" * 40), AIMessage(content="", tool_calls=[{"name": "x", "args": {"q": "b" * 30}, "id": "i"}])]
    base = _tokens(msgs)
    assert base >= 10 + 8  # 40 chars of question + the serialized call
    assert _tokens(msgs, overhead=400) == base + 100


# --- the fit -------------------------------------------------------------------------------

def test_small_conversation_is_untouched():
    msgs = _turn(0) + _turn(1)
    assert fit_to_budget(msgs, 6000) == msgs


def test_oldest_turns_are_dropped_first_and_the_current_one_is_kept():
    msgs = [m for n in range(6) for m in _turn(n, sizes=(1500, 1500))]  # ~6k chars each
    out = fit_to_budget(msgs, 4000)
    assert _tokens(out) <= 4000
    texts = [m.text for m in out if isinstance(m, HumanMessage)]
    assert texts[-1] == "q5" and "q0" not in texts  # oldest gone, current kept
    assert out[-len(_turn(5, sizes=(1500, 1500))):] == _turn(5, sizes=(1500, 1500))


def test_pairing_invariant_holds_for_every_budget():
    msgs = [m for n in range(6) for m in _turn(n, sizes=(900, 900, 900))]
    for budget in (1000, 1500, 2000, 3000, 6000, 20000):
        out = fit_to_budget(compact_history(msgs, 3), budget)
        assert find_dangling_tool_calls(out) == [], budget
        asked = {tc["id"] for m in out if isinstance(m, AIMessage) for tc in m.tool_calls}
        assert all(m.tool_call_id in asked for m in _tool_msgs(out)), budget


def test_current_turn_tool_results_are_truncated_largest_first_down_to_the_floor():
    cur = [HumanMessage("now")] + _turn(9, sizes=(3000, 600, 3000), answer=False)[1:]
    msgs = _turn(0, sizes=(200,)) + cur
    budget = 1500  # tokens = 6000 chars: needs cuts
    out = fit_to_budget(msgs, budget)
    assert _tokens(out) <= budget
    sizes = [len(m.content) for m in _tool_msgs(out) if m.tool_call_id.startswith("t9")]
    assert sizes[1] == 600  # the small one was not touched
    assert all(s >= TOOL_FLOOR_CHARS or s == 600 for s in sizes)
    cut = next(m for m in _tool_msgs(out) if len(m.content) < 3000 and m.tool_call_id.startswith("t9"))
    assert "[truncated" in cut.content and cut.content.startswith("rrr")
    assert out[-len(cur):][0].text == "now"  # the question itself is never cut


def test_gives_up_gracefully_when_even_the_floor_does_not_fit():
    cur = [HumanMessage("now")] + _turn(9, sizes=(5000,) * 6, answer=False)[1:]
    out = fit_to_budget(cur, 1000)  # 6 floors alone are 6000 chars > 4000
    assert all(len(m.content) <= TOOL_FLOOR_CHARS for m in _tool_msgs(out))
    assert find_dangling_tool_calls(out) == []
    assert len(_tool_msgs(out)) == 6  # nothing removed from the current turn


def test_overhead_counts_against_the_budget():
    msgs = _turn(0, sizes=(2000,)) + _turn(1, sizes=(2000,))
    assert _tokens(fit_to_budget(msgs, 2000)) <= 2000
    tight = fit_to_budget(msgs, 2000, overhead_chars=4000)
    assert _tokens(tight, 4000) <= 2000 and len(tight) < len(msgs)


def test_idempotent_and_input_not_mutated():
    msgs = [m for n in range(5) for m in _turn(n, sizes=(3000, 3000))]
    snapshot = [m.model_copy() for m in msgs]
    once = fit_to_budget(msgs, 2500)
    assert msgs == snapshot
    assert fit_to_budget(once, 2500) == once


def test_old_turn_without_answer_is_dropped_whole_and_stays_paired():
    msgs = _turn(0, sizes=(4000,), answer=False) + _turn(1, sizes=(4000,)) + _turn(2, sizes=(100,))
    out = fit_to_budget(msgs, 1200)
    assert find_dangling_tool_calls(out) == []
    assert out[0].text in {"q1", "q2"}


# --- wiring: the agent node ----------------------------------------------------------------

@tool
def big_list(n: int) -> str:
    """Return a bulky page."""
    return "x" * n


def test_tool_definition_chars_measures_the_bound_tools_once():
    assert 100 < tool_definition_chars([big_list]) < 1000
    assert tool_definition_chars([]) == 0


def _failing_scenario(budget):
    """Turn 1 stimulate-like result, turn 2 a 12k-char list page (the observed Groq 413)."""
    def reply(msgs):
        last = msgs[-1]
        question = [m for m in msgs if m.type == "human"][-1].text
        if last.type == "human":
            size = 3000 if question == "turn one" else 12000
            return call("big_list", {"n": size}, cid=f"c{len(msgs)}")
        return AIMessage(content=f"done {question}")

    model = fn_model(reply)
    graph = build_graph(model, [big_list], checkpointer=InMemorySaver(), keep_turns=3, context_budget=budget)

    async def run():
        cfg = {"configurable": {"thread_id": "t"}}
        for q in ("turn one", "turn two"):
            await graph.ainvoke({"messages": [HumanMessage(q)]}, config=cfg)
        return (await graph.aget_state(cfg)).values["messages"]

    stored = asyncio.run(run())
    return model, stored


def test_every_request_fits_the_budget_and_the_checkpoint_stays_whole():
    budget = 1500
    model, stored = _failing_scenario(budget)
    overhead = len(SYSTEM_PROMPT) + tool_definition_chars([big_list])
    sent = [_tokens([m for m in req if not isinstance(m, SystemMessage)], overhead) for req in model.seen]
    assert len(sent) == 4 and max(sent) <= budget, sent
    assert [len(m.content) for m in stored if isinstance(m, ToolMessage)] == [3000, 12000]  # untouched
    final = model.seen[-1]
    assert find_dangling_tool_calls(final) == []
    assert any("[truncated" in m.content for m in final if isinstance(m, ToolMessage))


def test_without_the_budget_the_same_scenario_overflows():
    model, _ = _failing_scenario(10**9)
    overhead = len(SYSTEM_PROMPT) + tool_definition_chars([big_list])
    assert max(_tokens([m for m in req if not isinstance(m, SystemMessage)], overhead) for req in model.seen) > 1500
