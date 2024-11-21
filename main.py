import os
#importaciones necesarias
from langchain.chains import ConversationalRetrievalChain

from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


#Cargar el archivo .env
_ = load_dotenv(find_dotenv())
#Configuramos el modelo de OpenAI
openai_api_key = os.environ["OPENAI_API_KEY"]
llm = OpenAI(temperature=0)

#Cargamos el archivo PDF
loader = PyPDFLoader('./Documentos/BOE-A-2024-24098.pdf')
loaded_data = loader.load()

print ("Número de páginas cargadas: ", len(loaded_data))


#Cortamos en trocitos la información (SPLIT)

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(loaded_data)
print("\n----------\n")
print ("Número de trozos: ", len(chunks))
print("\n----------\n")

#Los convertimos en numeritos (Embeddings) y guardamos en la base de datos
embeddings_model = OpenAIEmbeddings()

vector_db = Chroma.from_documents(chunks, embeddings_model)

#Preguntamos a la IA
question = "¿Qué dice el artículo 2?"

qa_chain = ConversationalRetrievalChain.from_llm(llm, vector_db.as_retriever())

history = []
response = qa_chain.invoke({"question": question, "chat_history": history})
history.append((question, response["answer"]))
print("\n----------\n")
print("Pregunta 1:", question)
print("\n----------\n")
print("Respuesta:", response["answer"])
print("\n----------\n")
