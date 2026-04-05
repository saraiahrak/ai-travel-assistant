import dspy
from dspy.evaluate import Evaluate
from schema import TravelRouter
from .router_data import devset, travel_metric # Import your unseen test data

# 1. Setup the LM
lm = dspy.LM('ollama_chat/qwen2.5:3b', api_base='http://localhost:11434')
dspy.settings.configure(lm=lm)

def run_evaluation():
    print("🧪 Starting Evaluation on Unseen Data...")

    # 2. Load the Optimized Program
    router = dspy.Predict(TravelRouter)
    try:
        router.load("dspy_modules/travel_router_v1.json")
    except FileNotFoundError:
        print("❌ Error: Optimized weights not found. Run train_router.py first.")
        return

    # 3. Setup the Evaluator
    # This will run the devset through the router and calculate the metric score.
    evaluator = Evaluate(
        devset=devset, 
        metric=travel_metric, 
        display_progress=True, 
        display_table=5 # Shows a neat table of the first 5 results
    )

    # 4. Run it!
    score = evaluator(router)
    print(f"\n✅ Final Evaluation Score: {score}%")
    
    if score > 80:
        print("🚀 The model has successfully generalized the routing logic!")
    else:
        print("⚠️ The model may need more diverse training examples.")

if __name__ == "__main__":
    run_evaluation()