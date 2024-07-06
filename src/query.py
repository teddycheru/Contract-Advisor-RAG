from retriever import load_contracts
from generator import chunk_contracts, create_docstore
from evaluator import load_evaluation_data, bleu, calculate_hallucination_score, calculate_additional_metrics
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load and chunk contracts
contracts = load_contracts()
documents = chunk_contracts(contracts)

# Create docstore
docstore = create_docstore(documents, openai_api_key)

# Load evaluation data
EVAL_FILE = "../data/qna/Robinson Q&A.docx"
evaluation_data = load_evaluation_data(EVAL_FILE)

# Function to perform similarity search
def query_function(query):
    results = docstore.similarity_search(query, k=5)
    return results[0].page_content if results else ""

# Prepare queries and references from evaluation data
queries = [qa['question'] for qa in evaluation_data]
references = [qa['answer'] for qa in evaluation_data]

# Print individual results and metrics
total_bleu_score = 0
total_hallucination_score = 0
num_samples = len(queries)

print("Evaluation Results:")
print("="*50)

for query, reference in zip(queries, references):
    generated_answer = query_function(query)
    bleu_score_value = bleu(generated_answer, reference)
    hallucination_score_value = calculate_hallucination_score(generated_answer, reference)
    additional_metrics = calculate_additional_metrics(generated_answer, reference)
    
    total_bleu_score += bleu_score_value
    total_hallucination_score += hallucination_score_value
    
    print(f"Query: {query}")
    print(f"Generated Answer: {generated_answer}")
    print(f"Reference Answer: {reference}")
    print(f"BLEU Score: {bleu_score_value:.2f}")
    print(f"Hallucination Score: {hallucination_score_value:.2f}")
    print("-"*50)

avg_bleu_score = total_bleu_score / num_samples
avg_hallucination_score = total_hallucination_score / num_samples

avg_metrics = {
    "average_bleu_score": avg_bleu_score,
    "average_hallucination_score": avg_hallucination_score
}

print("Average Metrics:")
print("="*50)
for metric, value in avg_metrics.items():
    print(f"{metric.replace('_', ' ').capitalize()}: {value:.2f}")

