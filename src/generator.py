from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def chunk_contracts(contracts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    documents = [Document(page_content=chunk) for contract in contracts for chunk in splitter.split_text(contract)]
    return documents

def create_docstore(documents, openai_api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    docstore = Chroma.from_documents(documents, embeddings)
    return docstore
