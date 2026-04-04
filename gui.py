import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from main import app  # Import your compiled LangGraph workflow

st.set_page_config(page_title="AI Travel Assistant", page_icon="✈️")

st.title("🌍 Smart Travel Planner")
st.markdown("---")

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Initial graph state
    st.session_state.graph_state = {
        "messages": [], 
        "trip_context": {}, 
        "external_data": "", 
        "location": "", 
        "next_step": ""
    }

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Where are we going?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Update Graph State
    st.session_state.graph_state["messages"].append(HumanMessage(content=prompt))

    # Run the Graph
    with st.spinner("Thinking..."):
        # This calls your LangGraph 'app'
        final_state = app.invoke(st.session_state.graph_state)
        
        # Get the last AI message
        response = final_state["messages"][-1].content
        
        # Update session state with the new graph state for context
        st.session_state.graph_state = final_state

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})