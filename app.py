import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import re

# Load API keys
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Function to clean Groq model responses
def clean_response(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# Persona system prompts
PERSONAS = {
    "Default": "You are a helpful assistant. Please respond to the user queries in a formal way.",
    "RoastBot": "You are RoastBot. Always respond with witty, sarcastic roasts no matter what the user says.",
    "ShakespeareBot": "You are ShakespeareBot. Respond to all queries in Shakespeare-style old English prose.",
    "Emoji Translator": "You are EmojiBot. Translate everything into emoji-speak.",
    "Creative": "You are a highly creative assistant who replies in imaginative, playful ways using metaphors."
}

# Function to generate context-aware responses
def generate_response(question, llm_name, chat_history, persona_prompt):
    model = ChatGroq(model=llm_name)

    # Build conversation
    messages = [
        SystemMessage(content=persona_prompt)
    ]

    # Add chat history
    for role, msg in chat_history:
        if role == "user":
            messages.append(HumanMessage(content=msg))
        elif role == "bot":
            messages.append(AIMessage(content=msg))

    # Add new user input
    messages.append(HumanMessage(content=question))

    # Invoke model
    response = model.invoke(messages)
    return response.content

# --- Streamlit UI ---
st.set_page_config(page_title="Persona AI Chatbot", layout="wide", initial_sidebar_state="expanded")
st.title("🤖 Persona AI Chatbot")

# Sidebar selectors
llm_model = st.sidebar.selectbox("Choose a model", [
    "deepseek-r1-distill-llama-70b",
    "moonshotai/kimi-k2-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct"
])
persona_choice = st.sidebar.selectbox("Choose a persona", list(PERSONAS.keys()))

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.chat_input("Say something...")

if user_input:
    # Store user input
    st.session_state.chat_history.append(("user", user_input))

    # Get persona-specific response
    bot_response = generate_response(
        user_input,
        llm_model,
        st.session_state.chat_history,
        PERSONAS[persona_choice]
    )
    bot_response = clean_response(bot_response)

    # Store bot response
    st.session_state.chat_history.append(("bot", bot_response))

# Render messages
for role, message in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)
