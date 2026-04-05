import dspy

# ---------------------------------------------------------
# 1. THE TRAINSET
# These examples specifically address:
# - Pronoun resolution ("there", "it")
# - Staying in the 'fetch_attractions' lane for follow-ups
# - Handling vague packing requests
# ---------------------------------------------------------
trainset = [
    # HURDLE: Follow-up questions about timing (The Thailand/June case)
    dspy.Example(
        context="Current Location: Thailand",
        query="Is it better to go there in June?",
        next_step="fetch_attractions",
        target_city="Thailand"
    ).with_inputs('context', 'query'),
    
    # HURDLE: Pronouns for packing
    dspy.Example(
        context="Current Location: Iceland",
        query="What should I bring for a trip there?",
        next_step="fetch_packing",
        target_city="Iceland"
    ).with_inputs('context', 'query'),
    
    # HURDLE: Vague queries with NO location (Should trigger None/Interrupt)
    dspy.Example(
        context="Current Location: None",
        query="What should I pack for my next trip?",
        next_step="fetch_packing",
        target_city="None"
    ).with_inputs('context', 'query'),

    # HURDLE: Distinguishing between 'New Ideas' vs 'Info on Current'
    dspy.Example(
        context="Current Location: Paris",
        query="Are there any cool festivals in July?",
        next_step="fetch_attractions",
        target_city="Paris"
    ).with_inputs('context', 'query'),

    # HURDLE: Switching locations mid-chat
    dspy.Example(
        context="Current Location: Paris",
        query="Actually, I've decided on Tokyo instead. What can I do there?",
        next_step="fetch_attractions",
        target_city="Tokyo"
    ).with_inputs('context', 'query'),

    # HURDLE: Switching locations mid-chat
    dspy.Example(
        context="Current Location: Berlin",
        query="Do I need a jacket there?",
        next_step="fetch_packing",
        target_city="Berlin"
    ).with_inputs('context', 'query'),

    # HURDLE: Switching locations mid-chat
    dspy.Example(
        context="Current Location: None",
        query="Let's go to Sydney. Any tips?",
        next_step="fetch_attractions",
        target_city="Sydney"
    ).with_inputs('context', 'query')
]

# ---------------------------------------------------------
# 2. THE DEVSET (Testing for Generalization)
# We use different cities to ensure the model doesn't just 
# learn 'Thailand' but learns the 'Logic of Context'.
# ---------------------------------------------------------
devset = [
    # Testing the 'Is it better in [Month]' logic for a new city
    dspy.Example(
        context="Current Location: Rome",
        query="What about visiting in October?",
        next_step="fetch_attractions",
        target_city="Rome"
    ).with_inputs('context', 'query'),
    
    # Testing the 'There' pronoun resolution for packing
    dspy.Example(
        context="Current Location: Berlin",
        query="Do I need a jacket there right now?",
        next_step="fetch_packing",
        target_city="Berlin"
    ).with_inputs('context', 'query'),

    # Testing a complete change of heart
    dspy.Example(
        context="Current Location: London",
        query="I'm bored of Europe, let's go to Sydney. Any tips?",
        next_step="fetch_attractions",
        target_city="Sydney"
    ).with_inputs('context', 'query'),

    # Testing a generic destination request
    dspy.Example(
        context="Current Location: None",
        query="Where is a good place for a honeymoon in May?",
        next_step="fetch_destinations",
        target_city="None"
    ).with_inputs('context', 'query')
]

# ---------------------------------------------------------
# 3. THE METRIC (Stays the same)
# ---------------------------------------------------------
def travel_metric(gold, pred, trace=None):
    intent_ok = gold.next_step == pred.next_step
    gold_city = str(gold.target_city).lower()
    pred_city = str(pred.target_city).lower()
    city_ok = (gold_city == pred_city)
    return intent_ok and city_ok