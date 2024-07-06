# evaluator.py

from docx import Document as DocxDocument
import spacy
from transformers import pipeline
import sacrebleu

# Function to ensure the spaCy model is downloaded
def ensure_spacy_model(model_name="en_core_web_sm"):
    try:
        spacy.load(model_name)
    except OSError:
        from subprocess import run
        run(f"python -m spacy download {model_name}", shell=True)

# Ensure the spaCy model is installed
ensure_spacy_model()

# Load spaCy model for entity recognition
nlp = spacy.load("en_core_web_sm")

# Load NLI model
nli_model = pipeline("text-classification", model="roberta-large-mnli")

# Function to read text from .docx files
def read_docx(file_path):
    doc = DocxDocument(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to load evaluation data
def load_evaluation_data(file_path):
    data = read_docx(file_path)
    qa_pairs = []
    lines = data.split('\n')
    current_question = None
    for line in lines:
        line = line.strip()
        if line.startswith('Q') and ':' in line:
            if current_question:
                qa_pairs.append(current_question)
            current_question = {"question": line.split(':', 1)[1].strip()}
        elif line.startswith('A') and ':' in line and current_question:
            current_question["answer"] = line.split(':', 1)[1].strip()
            qa_pairs.append(current_question)
            current_question = None
    return qa_pairs

# Function to calculate BLEU score using sacrebleu
def bleu(pred, ref):
    return sacrebleu.sentence_bleu(pred, [ref]).score

# Hallucination scoring functions
def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

def calculate_hallucination_score(generated_text, reference_text):
    # Extract entities
    gen_entities = extract_entities(generated_text)
    ref_entities = extract_entities(reference_text)
    
    # Calculate entity overlap score
    common_entities = set(gen_entities) & set(ref_entities)
    entity_score = 1 - (len(common_entities) / len(set(gen_entities)))
    
    # Calculate NLI entailment score
    nli_result = nli_model(f"premise: {reference_text} hypothesis: {generated_text}")
    entailment_score = nli_result[0]['score'] if nli_result[0]['label'] == 'ENTAILMENT' else 0
    
    # Combine scores
    combined_score = (entity_score + (1 - entailment_score)) / 2
    return combined_score

# Additional evaluation metrics (if needed for RAGAS)
def calculate_additional_metrics(generated_text, reference_text):
    # Implement additional evaluation metrics as needed
    pass

# Function to evaluate RAG system
def evaluate_rag_system(query_function, queries, references):
    total_bleu_score = 0
    total_hallucination_score = 0
    num_samples = len(queries)
    
    for query, reference in zip(queries, references):
        generated_answer = query_function(query)
        bleu_score_value = bleu(generated_answer, reference)
        hallucination_score_value = calculate_hallucination_score(generated_answer, reference)
        
        total_bleu_score += bleu_score_value
        total_hallucination_score += hallucination_score_value
    
    avg_bleu_score = total_bleu_score / num_samples
    avg_hallucination_score = total_hallucination_score / num_samples
    
    return {
        "average_bleu_score": avg_bleu_score,
        "average_hallucination_score": avg_hallucination_score
    }
