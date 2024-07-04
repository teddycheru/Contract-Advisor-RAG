import os
from dotenv import load_dotenv
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Function to read text from .docx files
def read_docx(file_path):
    doc = DocxDocument(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_contracts(contracts_dir):
    load_dotenv()
    contracts = []
    for filename in os.listdir(contracts_dir):
        if filename.endswith(".docx"):
            file_path = os.path.join(contracts_dir, filename)
            contracts.append(read_docx(file_path))
    return contracts

def chunk_contracts(contracts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    documents = [Document(page_content=chunk) for contract in contracts for chunk in splitter.split_text(contract)]
    return documents

if __name__ == "__main__":
    CONTRACTS_DIR = "../data/contracts"
    contracts = load_contracts(CONTRACTS_DIR)
    documents = chunk_contracts(contracts)
    print(f"Total documents chunked: {len(documents)}")
