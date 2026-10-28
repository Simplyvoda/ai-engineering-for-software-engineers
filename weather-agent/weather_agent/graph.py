"""The LangGraph wiring.

    START -> llm --(tool calls?)--> tools --+
              ^                             |
              +-----------------------------+
              |
              +--(no tool calls)--> compose_final -> END

The llm <-> tools loop lets the model chain calls: geocode first, then weather.
"""

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from .tools import TOOLS

AGENT_PROMPT = SystemMessage(
    content=(
        "You are a weather assistant. To answer a weather question, first call "
        "geocode_city for each city, then call current_weather with the returned "
        "latitude, longitude and city name. You can call tools for several cities "
        "at once. Never invent weather data. If a tool reports an error, accept it."
    )
)

COMPOSE_PROMPT = SystemMessage(
    content=(
        "Write the final answer for the user using only the tool results in this "
        "conversation. Give one short line per city. If a city could not be found "
        "or its data failed, say so plainly. If the user asked for Fahrenheit/imperial, "
        "convert the numbers. Do not call tools."
    )
)


def route_after_llm(state: MessagesState) -> str:
    """Go to the tools if the model asked for any, otherwise finish up."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "compose_final"


def build_graph(llm):
    """Build and compile the weather agent graph around any tool-capable chat model."""
    llm_with_tools = llm.bind_tools(TOOLS)

    def call_llm(state: MessagesState):
        response = llm_with_tools.invoke([AGENT_PROMPT] + state["messages"])
        return {"messages": [response]}

    def compose_final(state: MessagesState):
        response = llm.invoke([COMPOSE_PROMPT] + state["messages"])
        return {"messages": [response]}

    # handle_tool_errors=True: a ValueError/RuntimeError from a tool becomes an
    # error message the model can read, instead of crashing the whole graph.
    tool_node = ToolNode(TOOLS, handle_tool_errors=True)

    graph = StateGraph(MessagesState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", tool_node)
    graph.add_node("compose_final", compose_final)

    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", route_after_llm, {"tools": "tools", "compose_final": "compose_final"})
    graph.add_edge("tools", "llm")  # the loop
    graph.add_edge("compose_final", END)
    return graph.compile()
