# generator.py

import os
from dotenv import load_dotenv
import openai  # Correct import statement

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from retriever import chunk_contracts, load_contracts

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set API key directly to openai module

# Load and chunk contracts
CONTRACTS_DIR = "../data/contracts"
contracts = chunk_contracts(load_contracts(CONTRACTS_DIR))

# Create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

# Initialize and populate Chroma index
docstore = Chroma.from_documents(contracts, embeddings)
print(f"Documents indexed in Chroma: {len(docstore)}")
