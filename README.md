# Travel AI Assistant (DSPy + LangGraph)

A state-aware travel routing agent designed to handle complex multi-turn context and intent classification using local Large Language Models (**Ollama / Qwen 2.5 3B**). 

This project utilizes **DSPy** for programmatic optimization and **LangGraph** for robust state management. By moving away from manual "Prompt Engineering" toward a compiled approach, the agent achieves high accuracy in resolving ambiguous user queries and maintaining location context.

---

## Core Features

* **State-Aware Routing**: Resolves ambiguous pronouns (e.g., "there", "it") by cross-referencing LangGraph's persistent state with the user query.
* **Compiled Logic**: Uses serialized JSON weights to provide the 3B model with optimized instructions and few-shot examples.
* **Intent Specialization**: Dynamically routes queries between Weather, Packing, and Attractions specialists.
* **Human-in-the-loop**: Triggers an interrupt to request missing location data if a city cannot be resolved from the query or the conversation history.

---

## Project Structure

```text
travel-ai-assistant/
├── main.py                # Entry point: LangGraph orchestration
├── nodes.py               # Graph Logic (Router, Weather, Packing, Attractions)
├── schema.py              # DSPy Signatures and State definitions
├── dspy_modules/          # Compiled programs and optimized weights
│   └── travel_router_v1.json
└── optimizers/            # Optimization and Validation scripts
    ├── router_data.py     # Training (Trainset) and Testing (Devset) data
    ├── train_router.py    # Script to compile/optimize the Router
    └── test_router.py     # Script to evaluate accuracy on unseen data
```

---

## DSPy Optimization Workflow

Smaller language models often struggle with context retention in follow-up questions. This project implements a Serialized Optimization Workflow to mitigate these hurdles:

1. **Signature Definition**: Explicitly defined the boundaries for `fetch_packing` versus `fetch_attractions` within the TravelRouter signature in `schema.py`.
2. **Dataset Curation**: Created a trainset in `router_data.py` that specifically includes "context-heavy" examples (e.g., queries using pronouns like 'there' where the city is only present in the context field).
3. **BootstrapFewShot**: Leveraged the DSPy BootstrapFewShot optimizer. This teleprompter runs the signature against the training data, identifies successful traces, and selects the most effective few-shot examples to "prime" the Qwen 3B model.
4. **Compilation**: Saved the optimized prompt instructions and chosen examples into a JSON artifact (`travel_router_v1.json`) for production use.
5. **Production Loading**: The `router_node` loads these weights at runtime using `router.load()`, ensuring consistent performance and 100% routing accuracy without the overhead of re-running the optimizer during live chat.

---

## Installation and Usage

### 1. Environment Setup

Install the necessary dependencies:

```bash
pip install dspy-ai langgraph langsmith
```

### 2. Model Configuration

Ensure Ollama is running with the Qwen 2.5 3B model:

```bash
ollama run qwen2.5:3b
```

### 3. Optimization (The Training Phase)

To re-compile the router logic based on updated training data:

```bash
python -m optimizers.train_router
```

### 4. Validation (The Testing Phase)

Verify the model's accuracy on the held-out development set:

```bash
python -m optimizers.test_router
```

### 5. Execution

Run the assistant:

```bash
python main.py
```

---

## Performance Benchmarks (Dev Set)

The router was validated against a development set of queries the model did not see during the training phase. The test cases specifically targeted "pronoun resolution" and "tool-switching" hurdles.

* **Intent Detection**: 100% Accuracy  
* **Entity Resolution (Pronouns)**: 100% Accuracy  
* **Context Persistence**: 100% Accuracy  

---

## Implementation Logic

* **Context Fallback**: The `router_node` in `nodes.py` uses a dual-check system. If the LLM identifies a `target_city` as `None` (common in follow-up queries like "What about there?"), the node falls back to the location currently stored in the LangGraph `AgentState`.
* **State Updates**: Every successful routing update writes the new or existing location back to the global state, ensuring that even if the user switches topics, the "Current Location" remains accurate for the specialists.
* **Safety Guards**: The system utilizes a `NodeInterrupt` if neither the user query nor the current state can provide a valid target city, ensuring the specialist nodes never execute with empty data.
* **Module Resolution**: The project uses the `python -m` execution style to maintain clean imports between the root directory and the `optimizers/` sub-package, resolving standard Python pathing issues.