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

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from flybrain.agent_mcp import server_gone, stopped_error

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
        done = sum(
            1 for m in state["messages"][:-1]
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


def build_graph(model, tools):
    """Compile the explicit agent <-> tools loop for a chat model and tools."""
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
    return graph.compile()


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
