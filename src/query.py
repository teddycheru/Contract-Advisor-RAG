from retriever import load_contracts, chunk_contracts
from generator import create_embeddings, initialize_chroma_index
from evaluator import calculate_hallucination_score, calculate_additional_metrics, load_evaluation_data

# Paths to data directories and files
CONTRACTS_DIR = "../data/contracts"
EVAL_FILE = "../data/qna/Robinson Q&A.docx"

# Load contract data
contracts = load_contracts()

# Chunk contracts
documents = chunk_contracts(contracts)

# Create embeddings
embeddings = create_embeddings()

# Initialize Chroma index
docstore = initialize_chroma_index(documents, embeddings)

# Main function to run the query and calculate scores
def run_query(query, evaluation_data):
    results = docstore.similarity_search(query, k=5)

    for i, result in enumerate(results):
        generated_text = result.page_content
        reference_text = ""
        for qa_pair in evaluation_data:
            if query in qa_pair["question"]:
                reference_text = qa_pair["answer"]
                break
        
        hallucination_score = calculate_hallucination_score(generated_text, reference_text)
        
        print(f"Result {i + 1}:\n{generated_text[:500]}\n")
        print(f"Hallucination Score: {hallucination_score}\n")

if __name__ == "__main__":
    # Load evaluation data
    evaluation_data = load_evaluation_data(EVAL_FILE)
    
    # Example query
    query = "Who owns the IP?"
    run_query(query, evaluation_data)
