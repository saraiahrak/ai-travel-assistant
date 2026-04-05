import dspy
import uuid
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from travel_graph import create_graph

# --- SETUP ---
lm = dspy.LM('ollama_chat/qwen2.5:3b', api_base='http://localhost:11434', max_tokens=500)
dspy.settings.configure(lm=lm)

app = create_graph()

# Generate a unique session ID for this run
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

def run_cli():
    
# 1. WELCOME MESSAGE
    print("\n===============================================")
    print("🌍 Welcome to your AI Travel Assistant!")
    print("I can help with packing, attractions, and weather.")
    print("Type 'quit', 'exit', or 'q' to stop.")
    print("===============================================\n")

    while True:
        user_input = input("You: ").strip()

        # 2. QUIT OPTION
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nSafe travels! Goodbye. ✈️")
            break

        if not user_input:
            continue        
        # 1. Run the graph
        # If it hits a breakpoint, it will stop and return here
        app.invoke({"messages": [HumanMessage(content=user_input)]}, config)

        snapshot = app.get_state(config)
        
        # While the graph is "Waiting" at an interrupt
        while snapshot.next: 
            # If location is missing, prompt the user
            if snapshot.tasks and snapshot.tasks[0].interrupts:
                interrupt_msg = snapshot.tasks[0].interrupts[0].value
                print(f"\n--- [ACTION REQUIRED] ---\n{interrupt_msg}")
                
                user_response = input("> ").strip()
                app.invoke(Command(resume=user_response), config)
                snapshot = app.get_state(config)
            else:
                # Location exists, just tell the graph to finish
                app.invoke(None, config)
                snapshot = app.get_state(config)

        # Print the final AI response
        final_state = app.get_state(config)
        if final_state.values.get("messages"):
            print(f"\nAssistant: {final_state.values['messages'][-1].content}\n")

if __name__ == "__main__":
    run_cli()