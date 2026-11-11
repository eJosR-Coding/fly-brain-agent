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
import json
import math
import os
import re
import time
from collections.abc import AsyncIterator

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.prebuilt import ToolNode

from flybrain.agent_mcp import ServerStoppedError, server_gone, stopped_error

SYSTEM_PROMPT = (
    "You explore the FlyWire fruit-fly brain through tools. "
    "Start with search_neurons to find neurons, then act on the returned "
    "selection_id tickets with the other tools. Never invent neuron ids: only "
    "use ids and tickets the tools gave you. For wiring questions use neighbors, "
    "find_path, lesion_effect and top_hubs: they read the STRUCTURAL connectome "
    "(pairs with >= 5 synapses), so say 'structural connection', never claim signal flow "
    "or function from them; if they say the connectome is not built, tell the user. "
    "To ask what a group of neurons DOES, use preset_selection (published sets such as 'sugar' "
    "and 'mn9') and stimulate: it runs a whole-brain simulation (a leaky integrate-and-fire "
    "model with the Shiu et al. 2024 parameters). Always call its results a model's "
    "prediction, not the living fly (no neuromodulation, plasticity or nerve cord); it takes "
    "seconds, so run it once with sensible settings. If a result has a warning, relay it plainly. "
    "Format simulation answers the same way: the first line is a one-sentence verdict in bold; then "
    "a markdown table of the top responders (type, super class, Hz; no root ids; at most 5 rows), one per run when "
    "comparing; then the readout in one line; end with one short caveat line in italics and at most "
    "two next steps. One table per answer, never two tables side by side. Answer concisely."
)

MAX_STEPS = 6  # tool rounds (one round = the model asks for tools, the tools answer)
NO_ANSWER = "[no answer] the model returned an empty reply"


DEFAULT_KEEP_TURNS = 3  # user turns sent to the model in full (tool calls and results included)
MAX_TOOL_CHARS = 4000  # a single tool result in the sent window is cut to this many chars
_TRUNC_RESERVE = 40  # room for the "... [truncated N chars]" marker inside MAX_TOOL_CHARS


def keep_turns_from_env(env=None) -> int:
    """$FLYBRAIN_KEEP_TURNS (an int >= 1) or the default."""
    raw = (os.environ if env is None else env).get("FLYBRAIN_KEEP_TURNS")
    if raw is None:
        return DEFAULT_KEEP_TURNS
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"FLYBRAIN_KEEP_TURNS must be an integer >= 1, got {raw!r}")
    return value


def _truncate_tool_result(msg: ToolMessage, limit: int = MAX_TOOL_CHARS) -> ToolMessage:
    content = msg.content
    if not isinstance(content, str) or len(content) <= limit:
        return msg
    head = limit - _TRUNC_RESERVE
    marker = f"\n... [truncated {len(content) - head} chars]"
    return msg.model_copy(update={"content": content[:head] + marker})


def compact_history(messages, keep_turns: int = DEFAULT_KEEP_TURNS) -> list:
    """The message list to SEND to the model; the checkpointed thread is never changed.

    The last ``keep_turns`` user turns (the current one included, always) go in full, with any
    single tool result over MAX_TOOL_CHARS cut and marked. Older turns shrink to the question
    and the final answer text: their tool calls and tool results are dropped TOGETHER, so no
    tool call is left without its result (providers reject that). Pure and idempotent.
    """
    if isinstance(keep_turns, bool) or not isinstance(keep_turns, int) or keep_turns < 1:
        raise ValueError(f"keep_turns must be an int >= 1, got {keep_turns!r}")
    starts = [i for i, m in enumerate(messages) if m.type == "human"]
    if not starts:
        return list(messages)
    window_start = starts[-keep_turns:][0]
    out = list(messages[: starts[0]])  # anything before the first question (rare) stays
    bounds = [*starts, len(messages)]
    for a, b in zip(bounds, bounds[1:]):
        turn = messages[a:b]
        if a >= window_start:
            out.extend(_truncate_tool_result(m) if isinstance(m, ToolMessage) else m for m in turn)
            continue
        out.append(turn[0])
        answer = next(
            (m for m in reversed(turn[1:]) if isinstance(m, AIMessage)
             and not m.tool_calls and not m.invalid_tool_calls),
            None,
        )
        if answer is not None and turn[-1] is answer:  # only a turn that ENDED in an answer
            out.append(answer)
    return out


DEFAULT_CONTEXT_BUDGET = 6000  # tokens per request (Groq's free tier allows 8000 per minute)
MIN_CONTEXT_BUDGET = 1000
TOOL_FLOOR_CHARS = 1000  # budget fitting never cuts a tool result of the current turn below this


def context_budget_from_env(env=None) -> int:
    """$FLYBRAIN_CONTEXT_BUDGET (an int >= 1000 tokens) or the default."""
    raw = (os.environ if env is None else env).get("FLYBRAIN_CONTEXT_BUDGET")
    if raw is None:
        return DEFAULT_CONTEXT_BUDGET
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < MIN_CONTEXT_BUDGET:
        raise ValueError(
            f"FLYBRAIN_CONTEXT_BUDGET must be an integer >= {MIN_CONTEXT_BUDGET}, got {raw!r}"
        )
    return value


def _message_chars(msg) -> int:
    chars = len(msg.text)
    calls = [*(getattr(msg, "tool_calls", None) or []), *(getattr(msg, "invalid_tool_calls", None) or [])]
    if calls:
        chars += len(json.dumps(calls, default=str))
    return chars


def estimate_tokens(messages, overhead_chars: int = 0) -> int:
    """Rough token count: characters / 4 over the messages (tool-call args included) + overhead."""
    return math.ceil((sum(_message_chars(m) for m in messages) + overhead_chars) / 4)


def tool_definition_chars(tools) -> int:
    """Size of the tool definitions as the provider receives them (measured once per graph)."""
    return sum(len(json.dumps(convert_to_openai_tool(t))) for t in tools)


def fit_to_budget(messages, budget: int, overhead_chars: int = 0) -> list:
    """Shrink a (compacted) message list until its estimated tokens fit ``budget``.

    ``overhead_chars`` is what also gets sent besides the messages (system prompt + tool
    definitions). Steps: drop the oldest complete turn, whole (human, tool calls, results and
    answer together, so pairing stays valid), never the current one; then cut the largest tool
    results of the current turn, down to TOOL_FLOOR_CHARS each. If it still does not fit, the
    rest is sent as is. Pure and idempotent; the input is never mutated.
    """
    out = list(messages)
    over = lambda: estimate_tokens(out, overhead_chars) > budget  # noqa: E731
    while over():
        starts = [i for i, m in enumerate(out) if m.type == "human"]
        if len(starts) < 2:
            break
        del out[starts[0] : starts[1]]
    if not over():
        return out
    while over():
        cuttable = [
            (i, len(m.content)) for i, m in enumerate(out)
            if isinstance(m, ToolMessage) and isinstance(m.content, str) and len(m.content) > TOOL_FLOOR_CHARS
        ]
        if not cuttable:
            break
        i, size = max(cuttable, key=lambda c: c[1])
        excess = (estimate_tokens(out, overhead_chars) - budget) * 4
        out[i] = _truncate_tool_result(out[i], max(TOOL_FLOOR_CHARS, size - excess))
    return out


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


def build_graph(
    model, tools, checkpointer=None, keep_turns: int | None = None, context_budget: int | None = None
):
    """Compile the explicit agent <-> tools loop for a chat model and tools.

    Pass a checkpointer (e.g. InMemorySaver) to get multi-turn memory per thread_id. The thread
    is stored whole, but the model only sees ``compact_history`` of it (``keep_turns`` recent
    turns in full; default $FLYBRAIN_KEEP_TURNS or 3), then ``fit_to_budget`` keeps each request
    (system prompt + tool definitions + messages) under ``context_budget`` estimated tokens
    (default $FLYBRAIN_CONTEXT_BUDGET or 6000).
    """
    keep = keep_turns_from_env() if keep_turns is None else keep_turns
    budget = context_budget_from_env() if context_budget is None else context_budget
    overhead = len(SYSTEM_PROMPT) + tool_definition_chars(tools)
    model_with_tools = model.bind_tools(tools)

    async def call_model(state: MessagesState):
        fitted = fit_to_budget(compact_history(state["messages"], keep), budget, overhead)
        messages = [SystemMessage(SYSTEM_PROMPT), *fitted]
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


_QUOTA = re.compile(r"\b429\b|RESOURCE_EXHAUSTED|rate.?limit|quota|too many requests", re.I)
_TOO_LARGE = re.compile(r"request too large|\b413\b|payload too large", re.I)
_AUTH = re.compile(
    r"\b40[13]\b|PERMISSION_DENIED|UNAUTHENTICATED|invalid.{0,12}api.?key|api.?key.{0,20}(invalid|not valid|missing)"
    r"|incorrect api key|authentication",
    re.I,
)
SHORT_CAUSE = 120


def friendly_model_error(exc: BaseException, model_name: str | None = None) -> str | None:
    """A readable message for request-too-large, quota/rate-limit and auth failures of a provider, else None."""
    who = f"model {model_name}" if model_name else "the model"
    reason = " ".join(str(exc).split()) or type(exc).__name__
    status = next(
        (v for v in (getattr(exc, "status_code", None), getattr(exc, "code", None)) if isinstance(v, int)), None
    )
    if status == 413 or _TOO_LARGE.search(reason):  # before quota: Groq's 413 body says rate_limit_exceeded
        return (
            f"{who}: request too large for its per-minute token limit; "
            "start a new chat or ask a narrower question"
        )
    if status == 429 or _QUOTA.search(reason):
        cause = reason if len(reason) <= SHORT_CAUSE else reason[:SHORT_CAUSE] + "…"
        return f"{who} hit its quota or rate limit; switch model in the picker or retry later ({cause})"
    if status in (401, 403) or _AUTH.search(reason):
        return f"{who}: API key missing or invalid"
    if "model_not_found" in reason or (status == 404 and "model" in reason.lower()):
        return f"{who} is not available for this API key; pick another model in the picker"
    return None


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


def sum_usage(usages) -> dict:
    """Add up LangChain ``usage_metadata`` dicts; a field no call reported stays None (never 0).

    A call without ``total_tokens`` contributes input + output when both are known.
    """
    out = {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    for u in usages:
        if not u:
            continue
        known = {
            "input_tokens": u.get("input_tokens"),
            "output_tokens": u.get("output_tokens"),
            "total_tokens": u.get("total_tokens"),
        }
        if known["total_tokens"] is None and None not in (known["input_tokens"], known["output_tokens"]):
            known["total_tokens"] = known["input_tokens"] + known["output_tokens"]
        for key, value in known.items():
            if value is not None:
                out[key] = (out[key] or 0) + value
    return out


async def stream_events(
    graph,
    question: str,
    thread_id: str,
    max_steps: int = MAX_STEPS,
    turn_timeout: float | None = None,
    model_name: str | None = None,
    tool_extensions: dict[str, float] | None = None,
) -> AsyncIterator[dict]:
    """Run one turn and yield typed events (plain dicts) as the graph progresses.

    Types: ``tool_call`` {id, tool, args}, ``tool_result`` {id, tool, content, error, ms}
    (ms = since the matching tool_call), ``answer_delta`` {text} (token chunks of a reply that
    carries no tool call, only from models that stream; text before a tool call can show up too,
    so a consumer drops it on the next tool_call), ``answer`` {text} (the full text), ``error``
    {message}. Failures never raise: they become one final
    ``error`` event (same texts as ``ask``), so a consumer can always close the stream.
    ``thread_id`` selects the conversation memory (needs a checkpointer in the graph).
    A tool_call event is emitted when the model asks; if the budget then cuts the run,
    that last request never executes and the turn ends with the hard-stop error.
    ``turn_timeout`` (seconds) caps the whole turn: provider SDKs may retry 503s with
    backoff for minutes, so a slow turn ends with an ``error`` instead of silence.
    ``tool_extensions`` {tool name: seconds} adds that many seconds to the deadline whenever the
    model calls that tool (long experiments); other tools get no extra time.
    ``model_name`` names the model in friendlier quota / auth error texts.
    ``usage`` {calls, input_tokens, output_tokens, total_tokens} comes right before the final
    ``answer`` / ``error`` (only if the model was called): token counts are summed from every
    model call's usage_metadata and are None when the provider reported none.
    """
    _check_max_steps(max_steps)
    config = {
        "recursion_limit": 2 * max_steps + 2,
        "configurable": {"max_tool_rounds": max_steps, "thread_id": thread_id},
    }
    last_agent: AIMessage | None = None
    deadline = None if turn_timeout is None else time.monotonic() + turn_timeout
    updates = graph.astream(
        {"messages": [HumanMessage(question)]}, config=config, stream_mode=["updates", "messages"]
    ).__aiter__()
    started: dict[str, float] = {}
    failure: str | None = None  # set on every non-answer path; the thread is repaired below
    answered = False
    calls = 0
    call_usages: list = []  # one entry per model call: its usage_metadata, or None
    chunk_usage: list = []  # usage seen on streamed chunks of the call in flight (fallback)

    def usage_event() -> dict | None:
        return {"type": "usage", "calls": calls, **sum_usage(call_usages)} if calls else None

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
                mode, payload = update
                if mode == "messages":
                    chunk, meta = payload
                    if (isinstance(chunk, AIMessageChunk) and meta.get("langgraph_node") == "agent"
                            and chunk.usage_metadata):
                        chunk_usage.append(chunk.usage_metadata)
                    if (isinstance(chunk, AIMessageChunk) and meta.get("langgraph_node") == "agent"
                            and not chunk.tool_call_chunks and chunk.text):
                        yield {"type": "answer_delta", "text": chunk.text}
                    continue
                for node, delta in payload.items():
                    for msg in (delta or {}).get("messages", []):
                        if node == "agent":
                            last_agent = msg
                            calls += 1
                            # the finished message carries the summed usage; chunks are a fallback
                            call_usages.append(
                                getattr(msg, "usage_metadata", None)
                                or (sum_usage(chunk_usage) if chunk_usage else None)
                            )
                            chunk_usage = []
                            for tc in getattr(msg, "tool_calls", None) or []:
                                started[tc["id"]] = time.monotonic()
                                if deadline is not None and tool_extensions:
                                    deadline += tool_extensions.get(tc["name"], 0.0)
                                yield {"type": "tool_call", "id": tc["id"], "tool": tc["name"], "args": tc["args"]}
                        elif isinstance(msg, ToolMessage):
                            t0 = started.pop(msg.tool_call_id, None)
                            yield {
                                "type": "tool_result",
                                "id": msg.tool_call_id,
                                "tool": msg.name,
                                "content": msg.text,
                                "error": msg.status == "error",
                                "ms": 0 if t0 is None else round((time.monotonic() - t0) * 1000),
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
                failure = friendly_model_error(exc, model_name) or f"model call failed: {reason}"
        if failure is None:
            if last_agent is None:
                failure = NO_ANSWER
            elif getattr(last_agent, "tool_calls", None) or getattr(
                last_agent, "invalid_tool_calls", None
            ):
                failure = _hard_stop(max_steps)
        if failure is not None:
            if (spent := usage_event()) is not None:
                yield spent
            yield {"type": "error", "message": failure}  # first: a slow repair must not delay it
            return
        text = last_agent.text
        answered = True
        if (spent := usage_event()) is not None:
            yield spent
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
