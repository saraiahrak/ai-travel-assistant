import dspy
from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: List[BaseMessage]
    trip_context: dict
    external_data: str
    location: str
    is_interrupted: bool

# class TravelRouter(dspy.Signature):
#     """Analyze chat history. Identify if the user is asking about the current city or a new one."""
#     context = dspy.InputField(desc="History and currently known location.")
#     query = dspy.InputField(desc="User's latest request.")
#     next_step = dspy.OutputField(desc="fetch_destinations, fetch_attractions, fetch_packing, or general_chat")
#     target_city = dspy.OutputField(desc="The city being discussed.")


class TravelRouter(dspy.Signature):
    """Categorize travel queries and extract the destination city."""
    context = dspy.InputField(desc="Past messages and current city.")
    query = dspy.InputField(desc="User's current question.")
    
    # Use the desc to define the 'API' of your graph, not to 'prompt' the model.
    next_step = dspy.OutputField(desc="fetch_destinations, fetch_attractions, fetch_packing, or general_chat")
    target_city = dspy.OutputField(desc="The city mentioned or the one from context.")

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