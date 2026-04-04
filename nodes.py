import dspy
import requests
from langchain_core.messages import AIMessage
from schema import *

def router_node(state: AgentState):
    print("--- [ROUTING] ---")
    router = dspy.Predict(TravelRouter)
    pred = router(context=str(state["messages"][-3:]), query=state["messages"][-1].content)
    print(f"Decision: {pred.next_step} for {pred.target_city}")
    return {"location": pred.target_city, "next_step": pred.next_step}

def recommendation_node(state: AgentState):
    print("--- [SPECIALIST: DESTINATIONS] ---")
    spec = dspy.Predict(DestinationSpecialist)
    res = spec(context=str(state["messages"][-3:]), query=state["messages"][-1].content)
    return {"external_data": res.suggestions}

def attractions_node(state: AgentState):
    print("--- [SPECIALIST: ATTRACTIONS] ---")
    spec = dspy.Predict(AttractionSpecialist)
    res = spec(location=state.get("location", "Thailand"), 
               weather_context=state.get("external_data", ""), 
               query=state["messages"][-1].content)
    return {"external_data": res.activities}

def packing_node(state: AgentState):
    print("--- [SPECIALIST: PACKING] ---")
    spec = dspy.Predict(PackingSpecialist)
    res = spec(location=state.get("location", "Thailand"), 
               weather_context=state.get("external_data", ""), 
               query=state["messages"][-1].content)
    return {"external_data": res.list}

def weather_tool_node(state: AgentState):
    city = state.get("location", "London")
    print(f"--- [FETCHING WEATHER FOR: {city}] ---")
    try:
        search_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&format=json"
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