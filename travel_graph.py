from langgraph.graph import StateGraph, START, END
from schema import AgentState
from nodes import *

def create_graph():
    workflow = StateGraph(AgentState)

    # Register Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("fetch_weather", weather_tool_node)
    workflow.add_node("recommendation_node", recommendation_node)
    workflow.add_node("attractions_node", attractions_node)
    workflow.add_node("packing_node", packing_node)
    workflow.add_node("generate_response", assistant_node)

    # Define Edges
    workflow.add_edge(START, "router")

    workflow.add_conditional_edges(
        "router",
        lambda x: x["next_step"],
        {
            "fetch_destinations": "recommendation_node",
            "fetch_attractions": "fetch_weather",
            "fetch_packing": "fetch_weather",
            "general_chat": "generate_response"
        }
    )

    workflow.add_conditional_edges(
        "fetch_weather",
        lambda x: x["next_step"],
        {
            "fetch_attractions": "attractions_node",
            "fetch_packing": "packing_node"
        }
    )

    workflow.add_edge("recommendation_node", "generate_response")
    workflow.add_edge("attractions_node", "generate_response")
    workflow.add_edge("packing_node", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()