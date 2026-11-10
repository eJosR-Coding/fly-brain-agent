"""Hand-built LangGraph agent loop over the flybrain tools.

The loop, spelled out on purpose (no create_agent / create_react_agent):

    START -> agent -> (tool_calls?) -> tools -> agent -> ... -> END

1. "agent" node: the model sees the system prompt + conversation and either
   answers or asks for tools.
2. If the reply has tool_calls, "tools" executes them and appends ToolMessages.
3. Back to "agent". Stop when the model replies without tool_calls.

``recursion_limit`` is the emergency brake: a model stuck asking for tools
forever gets cut off and ``ask`` returns a readable message instead of crashing.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

SYSTEM_PROMPT = (
    "You explore the FlyWire fruit-fly brain through tools. "
    "Start with search_neurons to find neurons, then act on the returned "
    "selection_id tickets with the other tools. Never invent neuron ids: only "
    "use ids and tickets the tools gave you. Answer concisely."
)

MAX_STEPS = 12  # graph supersteps (each agent or tools turn counts as one)


def build_graph(model, tools):
    """Compile the explicit agent <-> tools loop for a chat model and tools."""
    model_with_tools = model.bind_tools(tools)

    async def call_model(state: MessagesState):
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [await model_with_tools.ainvoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    # handle_tool_errors=True: bad tool names/args come back as ToolMessages
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


async def ask(graph, question: str, max_steps: int = MAX_STEPS) -> str:
    """Run one question through the graph and return the final answer text."""
    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(question)]},
            config={"recursion_limit": max_steps},
        )
    except GraphRecursionError:
        return f"[hard stop] reached {max_steps} steps without a final answer"
    return result["messages"][-1].text
