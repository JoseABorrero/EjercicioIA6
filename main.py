import os
#importaciones necesarias
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

#Cargar el archivo .env
_ = load_dotenv(find_dotenv())
#Configuramos el modelo de OpenAI
openai_api_key = os.environ["OPENAI_API_KEY"]
chatModel = ChatOpenAI(model="gpt-4o-mini")

#Cargamos el archivo PDF
document = ""
loader = PyPDFLoader('./Documentos/BOE-A-2024-24098.pdf')
loaded_data = loader.load()


print ("Número de páginas cargadas: ", len(loaded_data))
for page in loaded_data:
    document += page.page_content
    #print(page.page_content)
    #print("\n----------\n")

#Cortamos en trocitos la información (SPLIT)

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
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

response = vector_db.similarity_search(question)

print("\n----------\n")
print("Respuesta:", response)
print("\n----------\n")
