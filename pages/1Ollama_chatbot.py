import streamlit as st
from langchain_community.llms import Ollama
import requests
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


st.title("💬 Ollama LLM Chatbot")
if "chatbot_config" not in st.session_state:
    st.session_state.chatbot_config = False

#Set config variables from SessionState

if "ollama_endpoint" in st.session_state and "model" in st.session_state:
    ollama_endpoint = st.session_state.ollama_endpoint
    model = st.session_state.model
    #Define model settings
    llm_model = Ollama(base_url=ollama_endpoint, model=model, temperature=0.0) 
    #Create a prompt for ChatBot configuration and memory
    prompt_template = """You are a chatbot based on a LLM, always try to answer question in the same language 
    of the user. In the chat history you are 'assistant' and the human is 'user'
    The chat history before the last question is:
    {chat_history}
    And the last question is:
    {question}
    """
    prompt = PromptTemplate(
                            input_variables=['chat_history', 'question'],
                            template=prompt_template,
                            )
    #Create the chain for LangChain
    chain = prompt | llm_model | StrOutputParser()
    st.session_state.chatbot_config = True
    st.text(f'This is a chatbot based on {st.session_state.model} Model')
    #print(f'Model set {model}')

if "messages" not in st.session_state and st.session_state.chatbot_config:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

### Write Message History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
    else:
        st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

## Generator for Streaming Tokens
def generate_response(question):
    response = chain.invoke({'chat_history':st.session_state['messages'],
                             'question':question})
    return (response)

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑‍💻").write(prompt)
    response = generate_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant", avatar="🤖").write(response)