import streamlit as st
import pathlib
import requests
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
import os
from utils.langchain_utils import response_rag
from utils.rag_utils import load_document



st.title("Chat with your PDF using RAG")
st.text(f'The model selected is {st.session_state.model}')

#Define embedding model
oll_embeddings = OllamaEmbeddings(base_url=st.session_state.ollama_endpoint,
                                 model='znbang/bge:large-en-v1.5-f16'
                                 )

documents_loaded = False

#Save uploaded file function
def save_uploaded_file(uploaded_file, path):
    # Verify destination folder, create folder if not exist
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return os.path.join(path, uploaded_file.name)

#Load the PDF to process
uploaded_file = st.file_uploader(label="Please upload your Pdf file to process", type=['pdf'])

if uploaded_file is not None and "db_created" not in st.session_state:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
        file_extension = uploaded_file.name[-3:]

    documents = load_document(tmp_file_path, file_extension)

    st.text('Document loading complete')
    documents_loaded = True
    # Split text 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                   chunk_overlap=100,
                                                   separators=["\n\n", "\n", " ", ""])
    splits = text_splitter.split_documents(documents)
    #print((splits[5].page_content))
    st.text('Document split complete')
    # Create a vectorstore
    st.session_state.vectorstore = Chroma.from_documents(documents=splits,
                                        embedding=oll_embeddings,
                                        #persist_directory="./chroma_db"
                                        )
    print('Vector chroma complete')
    st.session_state.db_created = True


if "messages_rag" not in st.session_state and "db_created" in st.session_state:
    st.session_state["messages_rag"] = [{"role": "assistant", "content": "How can I help you?"}]

### Write Message History
if "messages_rag" in st.session_state:
    for msg in st.session_state.messages_rag:
        if msg["role"] == "user":
            st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
        else:
            st.chat_message(msg["role"], avatar="🤖").write(msg["content"])


    if prompt := st.chat_input():
        st.session_state.messages_rag.append({"role": "user", "content": prompt})
        st.chat_message("user", avatar="🧑‍💻").write(prompt)
        response = response_rag(prompt)
        st.session_state.messages_rag.append({"role": "assistant", "content": response})
        st.chat_message("assistant", avatar="🤖").write(response)







