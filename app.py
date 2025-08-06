import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import re

# Load API keys from .env file
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Function to clean Groq model responses
def clean_response(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# Function to generate context-aware responses
def generate_response(question, llm_name, chat_history):
    model = ChatGroq(model=llm_name)

    # Build full conversation
    messages = [
        SystemMessage(content="You are a helpful assistant. Please respond to the user queries in a formal way.")
    ]

    # Add chat history to messages
    for role, msg in chat_history:
        if role == "user":
            messages.append(HumanMessage(content=msg))
        elif role == "bot":
            messages.append(AIMessage(content=msg))

    # Add the latest user question
    messages.append(HumanMessage(content=question))

    # Get model response
    response = model.invoke(messages)
    return response.content

# --- Streamlit UI Setup ---
st.set_page_config(page_title="AI Chatbot", layout="wide", initial_sidebar_state="expanded")
st.title("💬 Context-Aware AI Chatbot")

# Sidebar model selector
llm_model = st.sidebar.selectbox("Choose a model", [
    "deepseek-r1-distill-llama-70b",
    "moonshotai/kimi-k2-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct"
])

# Session state to keep chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input field at bottom
user_input = st.chat_input("Ask me anything...")

# Handle user input
if user_input:
    # Add user message
    st.session_state.chat_history.append(("user", user_input))

    # Generate bot response with context
    bot_response = generate_response(user_input, llm_model, st.session_state.chat_history)
    bot_response = clean_response(bot_response)

    # Add bot response
    st.session_state.chat_history.append(("bot", bot_response))

# Render chat messages
for role, message in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(message)
    elif role == "bot":
        st.chat_message("assistant").write(message)
