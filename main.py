import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.rate_limiters import InMemoryRateLimiter
import streamlit as st
from query_processor import QueryProcessor

load_dotenv()

rate_limiter = InMemoryRateLimiter(
    requests_per_second = 0.3,
    check_every_n_seconds = 0.1
)

queryProcessor = QueryProcessor(
    ChatMistralAI(model_name = os.environ['MISTRAL_LLM_MODEL'],temperature = 0.1, rate_limiter = rate_limiter), 
    verbose = True
    )

# Streamlit interface
st.title("Sports LLM")

# Initialize chat history as a list
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display existing chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Enter Your Query"):
    # Display user message in the chat interface
    with st.chat_message("user"):
        st.markdown(question)

    response = queryProcessor.processQuery(question, st.session_state.chat_history)

    # Append user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": question})

    # Append bot response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display bot message in the chat interface
    with st.chat_message("assistant"):
        st.markdown(response)