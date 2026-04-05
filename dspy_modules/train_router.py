import os
import sys
import dspy
from dspy.teleprompt import BootstrapFewShot
from schema import TravelRouter
from .router_data import trainset, travel_metric

# Get the directory where train_router.py lives
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the root directory to sys.path
sys.path.append(os.path.dirname(current_dir))
# Add the optimizers directory to sys.path
sys.path.append(current_dir)

# 1. Configure the local LM
lm = dspy.LM('ollama_chat/qwen2.5:3b', api_base='http://localhost:11434')
dspy.settings.configure(lm=lm)

def run_optimization():
    print("🚀 Starting DSPy Optimization...")

    # 2. Initialize the program
    router_program = dspy.Predict(TravelRouter)

    # 3. Setup the Optimizer (Teleprompter)
    # BootstrapFewShot finds the most helpful examples to insert into the prompt.
    optimizer = BootstrapFewShot(metric=travel_metric, max_bootstrapped_demos=3)

    # 4. Compile (The Training Step)
    optimized_router = optimizer.compile(router_program, trainset=trainset)

    # 5. Save the 'compiled' program to a folder
    folder = "dspy_modules"
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, "travel_router_v1.json")
    
    optimized_router.save(save_path)
    print(f"✅ Success! Optimized weights saved to: {save_path}")

if __name__ == "__main__":
    run_optimization()