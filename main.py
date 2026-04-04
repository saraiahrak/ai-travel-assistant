import dspy
from langchain_core.messages import HumanMessage
from travel_graph import create_graph

# --- SETUP ---
lm = dspy.LM('ollama_chat/qwen2.5:3b', api_base='http://localhost:11434', max_tokens=500)
dspy.settings.configure(lm=lm)

app = create_graph()

def run_cli():
    print("Welcome to your AI Travel Assistant! (Type 'quit' to exit)")
    state = {"messages": [], "trip_context": {}, "external_data": "", "location": "", "next_step": ""}
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]: break
        
        state["messages"].append(HumanMessage(content=user_input))
        state = app.invoke(state)
        print(f"\nAssistant: {state['messages'][-1].content}")

if __name__ == "__main__":
    run_cli()