import dspy
import requests
from langgraph.types import interrupt
from langchain_core.messages import AIMessage
from schema import *
from schema import TravelRouter

def router_node(state: AgentState):
    print("--- [ROUTING] ---")
    router = dspy.Predict(TravelRouter)
    router.load("dspy_modules/travel_router_v1.json")

    # 1. Capture the existing location from the state
    existing_loc = state.get("location")
    
    # 2. Run the Model
    response = router(
        context=f"Current Location: {existing_loc}\nHistory: {state['messages'][-3:]}",
        query=state["messages"][-1].content
    )

    # 3. THE RECOVERY LOGIC
    # Check if the model found a city in the query string itself
    model_city = response.target_city if response.target_city and str(response.target_city).lower() != "none" else None
    
    # Logic: 
    # - If the model found a city (e.g. "Thailand"), use it.
    # - If the model found NOTHING, but we already have a city in State, use State.
    # - Otherwise, it's truly None.
    final_city = model_city or existing_loc

    print(f"Decision: {response.next_step} for {final_city}")
    return {"location": final_city, "next_step": response.next_step}

def recommendation_node(state: AgentState):
    print("--- [SPECIALIST: DESTINATIONS] ---")
    spec = dspy.Predict(DestinationSpecialist)
    res = spec(context=str(state["messages"][-3:]), query=state["messages"][-1].content)
    return {"external_data": res.suggestions}

def attractions_node(state: AgentState):
    location = state.get("location")
    # If we get here and still no location, we simply return nothing.
    # The 'main.py' logic will ensure we don't proceed until location exists.
    if not location or location.lower() == "none":
        location = interrupt("Please provide a city for attractions.")

    print(f"--- [SPECIALIST: ATTRACTIONS for {location}] ---")
    spec = dspy.Predict(AttractionSpecialist)
    res = spec(
        location=location,
        weather_context=state.get("external_data", ""),
        query=state["messages"][-1].content
    )
    return {"location": location, "external_data": res.activities} # Return the actual list

def packing_node(state: AgentState):
    location = state.get("location")
    if not location or location.lower() == "none":
        location = interrupt("Please provide a city for packing.")

    print(f"--- [SPECIALIST: PACKING for {location}] ---")
    spec = dspy.Predict(PackingSpecialist)
    res = spec(
        location=location,
        weather_context=state.get("external_data", ""),
        query=state["messages"][-1].content
    )
    return {"location": location,"external_data": res.list}


def weather_tool_node(state: AgentState):
    location = state.get("location")
    if not location or location.lower() == "none":
        print("--- [PAUSING: Location Missing in Weather Node] ---")
        # This is the line that triggers the logic in main.py
        location = interrupt("I need a city to check the weather and packing requirements.")
    print(f"--- [FETCHING WEATHER FOR: {location}] ---")
    try:
        search_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&format=json"
        res = requests.get(search_url).json()['results'][0]
        w_url = f"https://api.open-meteo.com/v1/forecast?latitude={res['latitude']}&longitude={res['longitude']}&current_weather=true"
        temp = requests.get(w_url).json()['current_weather']['temperature']
        return {"external_data": f"Current weather in {res['name']}: {temp}°C."}
    except:
        return {"external_data": "Weather data unavailable."}

def assistant_node(state: AgentState):
    print("--- [GENERATING RESPONSE] ---")
    planner = dspy.ChainOfThought(TravelPlanner)
    data = state.get("external_data", "No specific data provided.")
    response = planner(context=str(state["messages"][-3:]), external_data=data, query=state["messages"][-1].content)
    return {"messages": [AIMessage(content=response.answer)], "external_data": ""}