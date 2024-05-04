from langchain_community.document_loaders import PyPDFLoader

def load_document(doc_path, file_extension):
    if file_extension=='pdf':
        loader = PyPDFLoader(doc_path)
        documents = loader.load()
        return(documents)

"""
loader = PyPDFLoader(tmp_file_path)
documents = loader.load()
st.text('Document loading complete')
documents_loaded = True
# Split text 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, 
                                                   chunk_overlap=40,
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
"""