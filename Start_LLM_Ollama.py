import streamlit as st
import requests

#Set session variables

if "ollama_is_running" not in st.session_state:
    st.session_state.ollama_is_running = False

# Define a empty list for available models
model_list = [] 

st.set_page_config(
    page_title='Start with LLms',
    page_icon='📋',
    layout='wide',
    initial_sidebar_state='collapsed'
)

st.title("Local LLM basics with Ollama")
st.text('We will configure Ollama to test LLMs running local in simply chatbot')

#Input Ollama server endpoint address

ollama_endpoint = st.text_input('Ollama address endpoint',
                                value='http://localhost:11434',
                                #value='https://fb53-2800-40-35-2ed-a540-84d4-4a9d-5e3e.ngrok-free.app',
                                help="Ollama API IP address")

st.session_state.ollama_endpoint = ollama_endpoint

#Select the model to use
#Request installed models
try:
    #Request to Ollama API to get models
    request_models = requests.get(ollama_endpoint+"/api/tags")
    if request_models.status_code==200:
        st.session_state.ollama_is_running = True        
except:
    st.session_state.ollama_is_running = False

if st.session_state.ollama_is_running:
    request_models = request_models.json()
#Save model names
    for i in range(len(request_models['models'])):
        model_list.append((request_models['models'][i]['name']))

model = st.selectbox("Select one LLM model to use", model_list) #Select box for model selection 

st.session_state.model = model

st.text('If you need install another please copy name below')

new_model = st.text_input('Paste the model name from Ollama Library, i.e. llama3')

if new_model and st.session_state.ollama_is_running:
    if st.button(label='Install'):
        parameters = {'name':new_model}
        api_endpoint = ollama_endpoint+"/api/pull"
        install_model = requests.post(url=api_endpoint, json=parameters)
        if install_model.status_code == 200:
            for chunk in install_model.iter_content(chunk_size=64):
                if chunk:
                    st.text(chunk)
    
if st.session_state.ollama_is_running:
    st.text(f'Ollama serve is running and the model selected is {st.session_state.model}')
    st.page_link("pages/1Ollama_chatbot.py", label='Go to Chatbot', icon='🤖')
    st.page_link("pages/2Chat_with_yourPDF.py", label='Go to Chat with your PDF', icon='🤖')