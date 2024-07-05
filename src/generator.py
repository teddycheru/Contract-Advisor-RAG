import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def create_embeddings():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return embeddings

def initialize_chroma_index(documents, embeddings):
    docstore = Chroma.from_documents(documents, embeddings)
    return docstore
