from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangSmith API Configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# Initialize Streamlit app
st.set_page_config(page_title="LLAMA 3.2 Chatbot", page_icon="🤖", layout="wide")

# Header Section
st.title("🤖 LLAMA 3.2 Chatbot")
st.markdown("""
<style>
.chat-bubble {
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    max-width: 70%;
    display: inline-block;
}
.user-bubble {
    background-color: #DCF8C6;
    align-self: flex-end;
}
.bot-bubble {
    background-color: #E1E1E1;
    align-self: flex-start;
}
.chat-container {
    display: flex;
    flex-direction: column;
}
</style>
""", unsafe_allow_html=True)

st.markdown("Welcome to the LLAMA 3.2 Chatbot powered by LangChain and Ollama. Ask me anything, and I'll do my best to help! ✨")

# Conversation History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input Section
st.markdown("### Ask a Question")
input_text = st.text_input("Type your question below 👇:")

# LLM Initialization
llm = OllamaLLM(model="llama3.2")  
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the queries."),
    ("user", "Question: {question}")
])
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Handle User Input
if input_text:
    try:
        response = chain.invoke({"question": input_text})
        # Add to conversation history
        st.session_state.chat_history.append(("user", input_text))
        st.session_state.chat_history.append(("bot", response))
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display Conversation History
st.markdown("### Conversation")
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f'<div class="chat-bubble user-bubble">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble bot-bubble">{message}</div>', unsafe_allow_html=True)

# Footer Section
st.markdown("---")
st.markdown("💡 **Pro Tip**: Try asking me to summarize text, solve math problems, or explain complex topics!")
st.markdown("🔗 [GitHub Repo](https://github.com/sushant-sinha1) | 🌟 Powered by LangChain, Ollama, and Streamlit")