import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama


#Define embedding model
"""oll_embeddings = OllamaEmbeddings(base_url=st.session_state.ollama_endpoint,
                                 model='snowflake-arctic-embed:latest',
                                 model_kwargs={"normalize_embeddings": True},
                                 )
"""
llm_model = Ollama(base_url=st.session_state.ollama_endpoint,
                   model=st.session_state.model,
                   temperature=0,
                   )

def search_info(query:str):
    # Create retriever
    #vectorstore = st.session_state.vectorstore
    retriever = st.session_state.vectorstore.as_retriever(search_type="similarity",
                                         search_kwargs={"k": 8})
    
    results = (retriever.invoke(query))
    print(results)
    return results 
    
        

def response_rag(query:str, context) ->str:
    # Create retriever
    #vectorstore = st.session_state.vectorstore
    retriever = st.session_state.vectorstore.as_retriever(search_type="similarity",
                                         search_kwargs={"k": 8 })
    
    # Define the prompt template with a placeholder for the question
    prompt_template = """
    You are an assistant for question-answering tasks.
    Use the provided context only to answer the following question: 
    <context>
    {context}
    </context>
    If do not information in the context please tell me. Always answer the question in the same language 
    as the question itself.
    Question: {question}

    Answer : 
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template= prompt_template,
        )
    #Create a chain 
    #print(query)
    chain = ({"context": retriever, "question": RunnablePassthrough()} 
            |prompt
            )
    print(chain.invoke(query)) 
    rag_chain = ({"context": retriever, "question": RunnablePassthrough()} 
                |prompt 
                |llm_model 
                | StrOutputParser()
                )
    return rag_chain.invoke(query)

