# query.py

from generator import embeddings
from langchain_community.vectorstores import Chroma

if __name__ == "__main__":
    query = "Who owns the IP? "

    # Ensure docstore is imported after it has been initialized in generator.py
    from generator import docstore

    # Perform similarity search using the Chroma index
    results = docstore.similarity_search(query, k=5)

    # Display the search results
    for i, result in enumerate(results):
        print(f"Result {i + 1}:\n{result.page_content[:500]}\n")
