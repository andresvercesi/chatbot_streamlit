#import torch
from langchain_community.llms import Ollama 
#from transformers import pipeline
#from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms.huggingface_pipeline  import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
import chardet
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import PyPDFLoader

#Define the model
llm_model = Ollama(base_url='https://cb3e-2800-40-35-2ed-b99e-fe39-e5c2-d051.ngrok-free.app', model='llama3')
print(llm_model.invoke('hola, puedes hablar en español?'))



# load text  
#loader = TextLoader("./data/abril.txt", autodetect_encoding=True)
loader = PyPDFLoader("C:\Dev\chatbot_streamlit\data\RENOVACION CF NICOLAS VERCESI CIA.MAPFRE.pdf")

documents = loader.load()
print('Document loading complete')

# Split text 

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
splits = text_splitter.split_documents(documents)
print(len(splits))
print('Document split complete')


# Load embedding model

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", 
encode_kwargs={"normalize_embeddings": True})
print('Embbeding model loading complete')

# Create a vectorstore
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
print('Vector chroma complete')

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print('Retriever created')


# Define the prompt template with a placeholder for the question

prompt_template = """
Eres un asistente virtual experto en polizas de seguro
que responde preguntas con ayuda del siguiente contexto: 
{context}
No debes inventar la respuesta. Si no la sabes debes decirlo. Cada vez que contestas
debes preguntar si tienen alguna otra pregunta.
Pregunta: {question}

Respuesta : 
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template= prompt_template,
)

# Create a prompt template
#prompt = ChatPromptTemplate.from_template(template)
print('Prompt created')

# Create a chain 

llm_chain = prompt | llm_model | StrOutputParser()
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain
print('Chain created')

print('Que quieres saber sobre tu poliza de seguro?')
while True:
    question = input('Algo mas? ')
    print(rag_chain.invoke(question))

#question = "En que juego se inspira el estanciero?"

#print(retriever.invoke(question))
#print(rag_chain.invoke(question))