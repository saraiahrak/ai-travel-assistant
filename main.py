print("SCRIPT STARTING...", flush=True)

import dspy
import requests
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# --- 1. SETUP DSPy (The Brain) ---
# Using Ollama locally
lm = dspy.LM('ollama_chat/qwen2.5:3b', api_base='http://localhost:11434', max_tokens=500)
dspy.settings.configure(lm=lm)

class TravelRouter(dspy.Signature):
    """
    Classify intent strictly:
    - 'fetch_packing': User asks what to WEAR, BRING, or PACK (e.g., 'What should I pack?', 'Do I need a coat?').
    - 'fetch_attractions': User asks what to DO, SEE, or VISIT (e.g., 'What are the landmarks?', 'Things to do').
    - 'fetch_destinations': User wants ideas for WHERE to go.
    - 'general_chat': Basic greetings or statements.
    """
    context = dspy.InputField(desc="Past conversation history including previous locations and topics")
    query = dspy.InputField(desc="Latest user message")
    
    next_step = dspy.OutputField(desc="fetch_destinations, fetch_attractions, fetch_weather, or general_chat")
    target_city = dspy.OutputField(desc="Inferred city from context if not mentioned.")

class TravelPlanner(dspy.Signature):
    """
    Expert Travel Assistant.
    - If 'external_data' is present AND relevant (e.g., packing, outdoor plans), use it.
    - If 'external_data' is empty or irrelevant to the query (e.g., just asking for history), 
      DO NOT mention weather or temperatures.
    - Focus on being a helpful guide, not just a weather reporter.
    """
    context = dspy.InputField()
    external_data = dspy.InputField()
    query = dspy.InputField()
    answer = dspy.OutputField(desc="A natural response that only mentions weather if it adds value.")

# --- 2. DEFINE THE STATE (The Memory) ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    trip_context: dict
    external_data: str
    next_step: str
    location: str

# --- 3. DEFINE THE NODES (The Nervous System) ---
def router_node(state: AgentState):
    print("--- [ROUTING] ---")
    router = dspy.Predict(TravelRouter)
    pred = router(context=str(state["messages"][-3:]), query=state["messages"][-1].content)
    
    # Debug to see what the LLM decided
    print(f"Decision: {pred.next_step} for {pred.target_city}")
    
    return {
        "location": pred.target_city,
        "next_step": pred.next_step 
    }
# --- 1. SPECIALIZED SIGNATURES ---
class DestinationSpecialist(dspy.Signature):
    """Suggest 3 travel destinations based on user tastes (beaches, hiking, etc.)."""
    context = dspy.InputField()
    query = dspy.InputField()
    suggestions = dspy.OutputField(desc="3 destinations with a brief 'why'.")

class AttractionSpecialist(dspy.Signature):
    """Suggest top landmarks and activities. Adjust suggestions if weather_context is provided."""
    location = dspy.InputField()
    weather_context = dspy.InputField(desc="Current weather/temp if available.")
    query = dspy.InputField()
    activities = dspy.OutputField(desc="Top 3 things to do.")

class PackingSpecialist(dspy.Signature):
    """Provide a packing list. You MUST use weather_context to decide on clothing."""
    location = dspy.InputField()
    weather_context = dspy.InputField(desc="Current weather/temp data.")
    query = dspy.InputField()
    list = dspy.OutputField(desc="A concise packing list tailored to the weather.")
# (Your existing TravelPlanner handles the Packing/Weather)

# --- 2. UPDATED NODES ---

# --- UPDATED SPECIALIST NODES ---
def recommendation_node(state: AgentState):
    print("--- [SPECIALIST: DESTINATIONS] ---")
    spec = dspy.Predict(DestinationSpecialist)
    res = spec(context=str(state["messages"][-3:]), query=state["messages"][-1].content)
    # Return string in external_data, DO NOT return a message
    return {"external_data": res.suggestions}

def attractions_node(state: AgentState):
    print("--- [SPECIALIST: ATTRACTIONS] ---")
    spec = dspy.Predict(AttractionSpecialist)
    res = spec(
        location=state.get("location", "Thailand"),
        weather_context=state.get("external_data", ""),
        query=state["messages"][-1].content
    )
    # Return string in external_data, DO NOT return a message
    return {"external_data": res.activities}

def packing_node(state: AgentState):
    print("--- [SPECIALIST: PACKING] ---")
    spec = dspy.Predict(PackingSpecialist)
    res = spec(
        location=state.get("location", "Thailand"),
        weather_context=state.get("external_data", ""),
        query=state["messages"][-1].content
    )
    # Return string in external_data, DO NOT return a message
    return {"external_data": res.list}

def assistant_node(state: AgentState):
    print("--- [GENERATING RESPONSE] ---")
    planner = dspy.ChainOfThought(TravelPlanner)
    
    # This now contains EITHER weather, recommendations, or attractions
    data_to_use = state.get("external_data", "No specific data provided.")
    
    response = planner(
        context=str(state["messages"][-3:]),
        external_data=data_to_use,
        query=state["messages"][-1].content
    )
    
    # Check if rationale exists before printing (Safety first!)
    if hasattr(response, 'rationale'):
        print(f"\x1b[34m[DEBUG Rationale]: {response.rationale}\x1b[0m")
    
    # Return the message and CLEAR the data so it doesn't leak into turn 2
    return {
        "messages": [AIMessage(content=response.answer)],
        "external_data": "" 
    }
# (The assistant_node now becomes the 'packing_node')

def weather_tool_node(state: AgentState):
    city_to_find = state.get("location", "London") # Default or extracted city
    print(f"--- [FETCHING WEATHER FOR: {city_to_find}] ---")
    
    try:
        # We search ONLY for the extracted city name. No more "Warsaw" ghosts!
        search_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_to_find}&count=1&format=json"
        res = requests.get(search_url).json()['results'][0]
        
        w_url = f"https://api.open-meteo.com/v1/forecast?latitude={res['latitude']}&longitude={res['longitude']}&current_weather=true"
        temp = requests.get(w_url).json()['current_weather']['temperature']
        
        # We return a fresh string that completely replaces the old one
        return {"external_data": f"Current weather in {res['name']}: {temp}°C."}
    except:
        return {"external_data": "Weather data unavailable."}
    

# --- 4. BUILD THE GRAPH ---
workflow = StateGraph(AgentState)

# Register all nodes
workflow.add_node("router", router_node)
workflow.add_node("fetch_weather", weather_tool_node)
workflow.add_node("recommendation_node", recommendation_node)
workflow.add_node("attractions_node", attractions_node)
workflow.add_node("packing_node", packing_node)
workflow.add_node("generate_response", assistant_node) # Final Polish

workflow.add_edge(START, "router")

# The Logic: Router -> Weather (Optional) -> Specialist -> Assistant
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

# ADD THIS: The "Dispatcher" from Weather to Specialists
workflow.add_conditional_edges(
    "fetch_weather",
    lambda x: x["next_step"],
    {
        "fetch_attractions": "attractions_node",
        "fetch_packing": "packing_node"
    }
)


# All specialists flow into the Assistant for a natural final word
workflow.add_edge("recommendation_node", "generate_response")
workflow.add_edge("attractions_node", "generate_response")
workflow.add_edge("packing_node", "generate_response")
workflow.add_edge("generate_response", END)

app = workflow.compile()

# --- 5. THE CLI INTERFACE ---
def run_cli():
    print("Welcome to your AI Travel Assistant! (Type 'quit' to exit)")
    # Initialize full state
    state = {"messages": [], "trip_context": {}, "external_data": "", "location": "", "next_step": ""}
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]: break
        
        state["messages"].append(HumanMessage(content=user_input))
        
        # Run the graph and update the WHOLE state object
        state = app.invoke(state)
        
        print(f"\nAssistant: {state['messages'][-1].content}")

if __name__ == "__main__":
    run_cli()