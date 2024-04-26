import streamlit as st
from langchain_community.llms import Ollama
import requests
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


#Input Ollama server endpoint address
ollama_endpoint = st.sidebar.text_input('Ollama address endpoint',
                                        value='http://localhost:11434',
                                        help="Ollama API IP address") 

# Define a empty list for available models
model_list = [] 

#Select the model to use
#Request installed models
request_models = requests.get(ollama_endpoint+"/api/tags") #Request to Ollama API to get models
if request_models.status_code==200:
    request_models = request_models.json()
#Save model names
    for i in range(len(request_models['models'])):
        model_list.append((request_models['models'][i]['name']))

model = st.sidebar.selectbox("Select Model", model_list) #Select box for model selection
if model and ('model_set' not in st.session_state):
    #Define model settings
    llm_model = Ollama(base_url=ollama_endpoint, model=model) 
    #Create a prompt for ChatBot configuration and memory
    prompt_template = """You are a chatbot based on a LLM, always try to answer in the same language.
    In the chat history you are 'assistan' an the human is 'user'
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
    st.session_state.model_set = True
    print('Model set')

st.title("💬 Ollama LLM Chatbot")

if "messages" not in st.session_state:
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
    st.session_state.messages.append({"role": "assistant", "content": response})
    return (response)

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑‍💻").write(prompt)
    st.session_state["full_message"] = ""
    st.chat_message("assistant", avatar="🤖").write(generate_response(prompt))
    st.session_state.messages.append({"role": "assistant", "content": st.session_state["full_message"]}) 
    