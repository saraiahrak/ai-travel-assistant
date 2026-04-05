from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from schema import AgentState
from nodes import *

memory = MemorySaver()
def create_graph():
    workflow = StateGraph(AgentState)

    # Register Nodes (Stay the same)
    workflow.add_node("router", router_node)
    workflow.add_node("fetch_weather", weather_tool_node)
    workflow.add_node("recommendation_node", recommendation_node)
    workflow.add_node("attractions_node", attractions_node)
    workflow.add_node("packing_node", packing_node)
    workflow.add_node("generate_response", assistant_node)

    workflow.add_edge(START, "router")

    # Routing from Router
    # travel_graph.py

    workflow.add_conditional_edges(
        "router",
        lambda x: x.get("next_step", "general_chat"), # Safely handle missing keys
        {
            "fetch_destinations": "recommendation_node",
            "fetch_attractions": "fetch_weather",
            "fetch_packing": "fetch_weather",
            "general_chat": "generate_response"
        }
    )
    workflow.add_conditional_edges(
        "fetch_weather",
        lambda x: x.get("next_step", "general_chat"), # Use .get() here too
        {
            "fetch_attractions": "attractions_node",
            "fetch_packing": "packing_node",
            "general_chat": "generate_response" # Add this as a fallback
        }
    )


    # SIMPLE EDGES: No more check_specialist_output loop!
    # Once the specialist is done, go straight to response.
    workflow.add_edge("recommendation_node", "generate_response")
    workflow.add_edge("attractions_node", "generate_response")
    workflow.add_edge("packing_node", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile(
        checkpointer=memory,
        # THE INTERRUPT: Hard stop before weather or specialists
        interrupt_before=["fetch_weather", "packing_node", "attractions_node"]
    )

# 1. Routing FROM Specialists
def check_specialist_output(state: AgentState):
    # If the specialist flagged an error, go back to router
    if state.get("next_step") == "error_recovery":
        print("⚠️ Missing Location! Re-routing to Router for extraction...")
        return "re-extract"
    return "finalize"