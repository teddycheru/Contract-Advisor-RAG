import os
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CONTRACTS_DIR = "../data/contracts"

def read_docx(file_path):
    doc = DocxDocument(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_contracts():
    contracts = []
    for filename in os.listdir(CONTRACTS_DIR):
        if filename.endswith(".docx"):
            file_path = os.path.join(CONTRACTS_DIR, filename)
            contracts.append(read_docx(file_path))
    return contracts

def chunk_contracts(contracts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    documents = [Document(page_content=chunk) for contract in contracts for chunk in splitter.split_text(contract)]
    return documents
