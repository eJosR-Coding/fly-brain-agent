"""Hand-built LangGraph agent loop over the flybrain tools.

The loop, spelled out on purpose (no create_agent / create_react_agent):

    START -> agent -> (tool_calls?) -> tools -> agent -> ... -> END

1. "agent" node: the model sees the system prompt + conversation and either
   answers or asks for tools.
2. If the reply has tool_calls, "tools" executes them and appends ToolMessages.
3. Back to "agent". Stop when the model replies without tool_calls.

``max_steps`` is the emergency brake, counted in TOOL ROUNDS (one agent -> tools trip):
a model stuck asking for tools forever gets cut off and ``ask`` returns a readable message
instead of crashing. ``max_steps=1`` allows one tool round plus the final answer.

Hardening: a malformed tool call is fed back to the model so it can retry, an empty
final reply is reported instead of returned as '', and a dead MCP server stops the run
with ServerStoppedError instead of letting the model improvise without tools.
"""

import asyncio
import time
from collections.abc import AsyncIterator

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from flybrain.agent_mcp import ServerStoppedError, server_gone, stopped_error

SYSTEM_PROMPT = (
    "You explore the FlyWire fruit-fly brain through tools. "
    "Start with search_neurons to find neurons, then act on the returned "
    "selection_id tickets with the other tools. Never invent neuron ids: only "
    "use ids and tickets the tools gave you. Answer concisely."
)

MAX_STEPS = 6  # tool rounds (one round = the model asks for tools, the tools answer)
NO_ANSWER = "[no answer] the model returned an empty reply"


def _tool_error_handler(exc: Exception) -> str:
    """ToolNode error handler: tool mistakes go back to the model, a dead server stops the run."""
    dead = server_gone(exc)
    if dead is not None:
        raise stopped_error(dead) from exc
    return f"Error: {exc!r}\n Please fix your mistakes."


def _route(state: MessagesState, config: RunnableConfig) -> str:
    """After the model: run tools, send malformed calls back, or finish.

    Enforces the tool-round budget (config["configurable"]["max_tool_rounds"], set by ask):
    when the model asks for yet another round past the budget, we stop instead of running it.
    """
    last = state["messages"][-1]
    wants = (
        "tools" if getattr(last, "tool_calls", None)
        else "fix_calls" if getattr(last, "invalid_tool_calls", None)
        else None
    )
    if wants is None:
        return END
    budget = (config.get("configurable") or {}).get("max_tool_rounds")
    if budget is not None:
        history = state["messages"][:-1]
        # budget is per TURN: with a checkpointer the history spans the whole conversation
        turn_start = max((i for i, m in enumerate(history) if m.type == "human"), default=-1) + 1
        done = sum(
            1 for m in history[turn_start:]
            if getattr(m, "tool_calls", None) or getattr(m, "invalid_tool_calls", None)
        )
        if done >= budget:
            return END  # ask() sees the unanswered tool request and reports the hard stop
    return wants


def _reject_invalid_calls(state: MessagesState):
    """Answer each malformed tool call with an error ToolMessage so the model can retry."""
    last = state["messages"][-1]
    replies = []
    for i, bad in enumerate(last.invalid_tool_calls):
        replies.append(
            ToolMessage(
                content=f"Error: could not use tool call {bad.get('name')!r}: "
                f"{bad.get('error') or 'malformed arguments'}. Send valid JSON arguments.",
                tool_call_id=bad.get("id") or f"invalid_call_{i}",
                name=bad.get("name") or "unknown",
                status="error",
            )
        )
    return {"messages": replies}


def build_graph(model, tools, checkpointer=None):
    """Compile the explicit agent <-> tools loop for a chat model and tools.

    Pass a checkpointer (e.g. InMemorySaver) to get multi-turn memory per thread_id.
    """
    model_with_tools = model.bind_tools(tools)

    async def call_model(state: MessagesState):
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [await model_with_tools.ainvoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=_tool_error_handler))
    graph.add_node("fix_calls", _reject_invalid_calls)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", _route, {"tools": "tools", "fix_calls": "fix_calls", END: END}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("fix_calls", "agent")
    return graph.compile(checkpointer=checkpointer)


def _hard_stop(max_steps: int) -> str:
    rounds = f"{max_steps} tool round{'s' if max_steps != 1 else ''}"
    return f"[hard stop] reached {rounds} without a final answer"


async def ask(graph, question: str, max_steps: int = MAX_STEPS) -> str:
    """Run one question through the graph and return the final answer text.

    ``max_steps`` = max tool rounds, enforced by the router (one round = agent + tools).
    recursion_limit is only a backstop: n rounds = 2n graph steps, the answer is 1 more and
    langgraph's finish is 1 more.
    Raises ServerStoppedError if the MCP server dies mid-run.
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError(f"max_steps must be an int >= 1, got {max_steps!r}")
    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(question)]},
            config={
                "recursion_limit": 2 * max_steps + 2,
                "configurable": {"max_tool_rounds": max_steps},
            },
        )
    except GraphRecursionError:
        return _hard_stop(max_steps)
    last = result["messages"][-1]
    if getattr(last, "tool_calls", None) or getattr(last, "invalid_tool_calls", None):
        return _hard_stop(max_steps)  # the model wanted another round past the budget
    answer = result["messages"][-1].text
    return answer if answer.strip() else NO_ANSWER


def _check_max_steps(max_steps) -> None:
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError(f"max_steps must be an int >= 1, got {max_steps!r}")


def find_dangling_tool_calls(messages) -> list[dict]:
    """Tool calls (``{id, name}``) in ``messages`` that no ToolMessage answers, in order."""
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    dangling = []
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        for tc in [*(m.tool_calls or []), *(m.invalid_tool_calls or [])]:
            if tc.get("id") and tc["id"] not in answered:
                dangling.append({"id": tc["id"], "name": tc.get("name") or "unknown"})
    return dangling


async def close_dangling_tool_calls(graph, config, reason: str) -> None:
    """Pair every unanswered tool call in the thread with an error ToolMessage.

    A turn cut mid-flight (hard stop, timeout, error) leaves an AIMessage whose tool_calls
    have no results; providers reject such a history, which would break the conversation for
    good. Without a checkpointer there is no thread to repair.
    """
    if getattr(graph, "checkpointer", None) is None:
        return
    state = await graph.aget_state(config)
    dangling = find_dangling_tool_calls((state.values or {}).get("messages", []))
    if not dangling:
        return
    closing = [
        ToolMessage(
            content=f"cancelled: the turn stopped before this tool ran ({reason})",
            tool_call_id=c["id"],
            name=c["name"],
            status="error",
        )
        for c in dangling
    ]
    # as_node="agent": the update is routed like a model reply, _route sees a ToolMessage
    # last and goes to END, so the next turn starts cleanly at the agent.
    await graph.aupdate_state(config, {"messages": closing}, as_node="agent")


async def stream_events(
    graph,
    question: str,
    thread_id: str,
    max_steps: int = MAX_STEPS,
    turn_timeout: float | None = None,
) -> AsyncIterator[dict]:
    """Run one turn and yield typed events (plain dicts) as the graph progresses.

    Types: ``tool_call`` {tool, args}, ``tool_result`` {tool, content, error},
    ``answer`` {text}, ``error`` {message}. Failures never raise: they become one final
    ``error`` event (same texts as ``ask``), so a consumer can always close the stream.
    ``thread_id`` selects the conversation memory (needs a checkpointer in the graph).
    A tool_call event is emitted when the model asks; if the budget then cuts the run,
    that last request never executes and the turn ends with the hard-stop error.
    ``turn_timeout`` (seconds) caps the whole turn: provider SDKs may retry 503s with
    backoff for minutes, so a slow turn ends with an ``error`` instead of silence.
    """
    _check_max_steps(max_steps)
    config = {
        "recursion_limit": 2 * max_steps + 2,
        "configurable": {"max_tool_rounds": max_steps, "thread_id": thread_id},
    }
    last_agent: AIMessage | None = None
    deadline = None if turn_timeout is None else time.monotonic() + turn_timeout
    updates = graph.astream(
        {"messages": [HumanMessage(question)]}, config=config, stream_mode="updates"
    ).__aiter__()
    failure: str | None = None  # set on every non-answer path; the thread is repaired below
    answered = False
    try:
        try:
            while True:
                left = None if deadline is None else max(deadline - time.monotonic(), 0)
                try:
                    update = await asyncio.wait_for(updates.__anext__(), left)
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    if deadline is None or time.monotonic() < deadline:
                        raise  # raised by a tool/provider itself, not our deadline: ordinary failure
                    failure = (
                        f"the turn took longer than {turn_timeout:g} s (the model provider "
                        "may be overloaded); try again or switch --model"
                    )
                    break
                for node, delta in update.items():
                    for msg in (delta or {}).get("messages", []):
                        if node == "agent":
                            last_agent = msg
                            for tc in getattr(msg, "tool_calls", None) or []:
                                yield {"type": "tool_call", "tool": tc["name"], "args": tc["args"]}
                        elif isinstance(msg, ToolMessage):
                            yield {
                                "type": "tool_result",
                                "tool": msg.name,
                                "content": msg.text,
                                "error": msg.status == "error",
                            }
        except GraphRecursionError:
            failure = _hard_stop(max_steps)
        except ServerStoppedError as exc:
            failure = str(exc)
        except Exception as exc:  # model/provider failure: report it, never hang the stream
            dead = server_gone(exc)
            if dead is not None:
                failure = str(stopped_error(dead))
            else:
                reason = " ".join(str(exc).split()) or type(exc).__name__
                failure = f"model call failed: {reason}"
        if failure is None:
            if last_agent is None:
                failure = NO_ANSWER
            elif getattr(last_agent, "tool_calls", None) or getattr(
                last_agent, "invalid_tool_calls", None
            ):
                failure = _hard_stop(max_steps)
        if failure is not None:
            await _repair(graph, config, updates, failure)
            yield {"type": "error", "message": failure}
            return
        text = last_agent.text
        answered = True
        yield {"type": "answer", "text": text if text.strip() else NO_ANSWER}
    finally:
        if not answered:
            # Runs on every exit that is not a clean answer: failure (after the error event),
            # CancelledError (task cancelled), GeneratorExit (aclose() when the client left).
            # The repair is shielded so a cancellation in flight cannot abort it halfway, and
            # the cancellation itself is never swallowed (this finally just re-raises it).
            await asyncio.shield(_repair(graph, config, updates, failure or "client disconnected"))


async def _repair(graph, config, updates, reason: str) -> None:
    """Stop the still-running graph stream, then pair any unanswered tool call."""
    try:
        await updates.aclose()
    except Exception:  # noqa: BLE001 - best effort; the repair below is what matters
        pass
    await close_dangling_tool_calls(graph, config, reason)
