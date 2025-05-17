import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()


# Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGSMITH_TRACING'] = "true"
os.environ['LANGSMITH_PROJECT'] = "Q&A Chatbot with Ollama"


## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries"),
        ("user", "Question: {question}")
    ]
)

def generate_responce(question:str, llm:str, temperature:float):
    # Load Ollama Model
    llm = OllamaLLM(model=llm, temperature=temperature)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer


# App Titile
st.title("Q&A Chatbot with Ollama LLM")

# Sidebar -> Settings
st.sidebar.title("Settings")

# Dropdown to select different types of Ollama LLM Models
llm = st.sidebar.selectbox(
    "Select Ollama LLM Model",
    [
        "gemma2:2b", "gemma3:1b", "gemma3", "gemma3:12b", "gemma3:27b",
        "deepseek-r1", "deepseek-r1:671b",
        "llama4:scout", "llama4:maverick",
        "llama3.3", "llama3.2", "llama3.2:1b", "llama3.2-vision", "llama3.2-vision:90b",
        "llama3.1", "llama3.1:405b",
        "qwq", "phi4", "phi4-mini",
        "mistral", "moondream", "neural-chat", "starling-lm", "codellama",
        "llama2-uncensored", "llava", "granite3.3"
    ],
)

# Adjust reponce parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)

# User Interface for user input
st.write("Ask any question")
user_input = st.text_input("Ask me")

if user_input:
    response = generate_responce(user_input, llm, temperature)
    st.write(response)
else:
    st.write("Question not asked")
