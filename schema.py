import dspy
from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: List[BaseMessage]
    trip_context: dict
    external_data: str
    next_step: str
    location: str

class TravelRouter(dspy.Signature):
    """Classify intent strictly: fetch_packing, fetch_attractions, fetch_destinations, or general_chat."""
    context = dspy.InputField(desc="Past conversation history")
    query = dspy.InputField(desc="Latest user message")
    next_step = dspy.OutputField(desc="Selection: fetch_destinations, fetch_attractions, fetch_packing, or general_chat")
    target_city = dspy.OutputField(desc="Inferred city/country.")

class TravelPlanner(dspy.Signature):
    """Expert Travel Assistant using external_data to provide natural responses."""
    context = dspy.InputField()
    external_data = dspy.InputField()
    query = dspy.InputField()
    answer = dspy.OutputField(desc="Natural response.")

class DestinationSpecialist(dspy.Signature):
    """Suggest 3 travel destinations."""
    context = dspy.InputField()
    query = dspy.InputField()
    suggestions = dspy.OutputField(desc="3 ideas with 'why'.")

class AttractionSpecialist(dspy.Signature):
    """Suggest top landmarks/activities."""
    location = dspy.InputField()
    weather_context = dspy.InputField()
    query = dspy.InputField()
    activities = dspy.OutputField(desc="Top 3 things to do.")

class PackingSpecialist(dspy.Signature):
    """Provide a packing list based on weather."""
    location = dspy.InputField()
    weather_context = dspy.InputField()
    query = dspy.InputField()
    list = dspy.OutputField(desc="Concise packing list.")