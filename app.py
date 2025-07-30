import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import re
# Load API keys
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# System prompt template
prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant. Please respond to the user queries in a formal way."),
    ("user", "Question: {question}")
])
def clean_response(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# Function to generate answer
def generate_response(question, llm):
    model = ChatGroq(model=llm)
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"question": question})

# UI
st.set_page_config(page_title="AI Chatbot", layout="wide",initial_sidebar_state="expanded")
st.title("💬 QnA Chatbot")

# Sidebar for model
llm_model = st.sidebar.selectbox("Select a model", [
    "deepseek-r1-distill-llama-70b",
    "moonshotai/kimi-k2-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct"
])

# Store messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box at bottom
with st.container():
    user_input = st.chat_input("Ask me anything...")

if user_input:
    # Add user input
    st.session_state.chat_history.append(("user", user_input))

    # Generate bot response
    bot_response = generate_response(user_input, llm_model)
    bot_response=clean_response(bot_response)
    st.session_state.chat_history.append(("bot", bot_response))

# Display chat history
for speaker, msg in st.session_state.chat_history:
    if speaker == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
