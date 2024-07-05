import spacy
from transformers import pipeline

nlp = spacy.load("en_core_web_sm")
nli_model = pipeline("text-classification", model="roberta-large-mnli")

def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

def calculate_hallucination_score(generated_text, reference_text):
    gen_entities = extract_entities(generated_text)
    ref_entities = extract_entities(reference_text)
    
    common_entities = set(gen_entities) & set(ref_entities)
    entity_score = 1 - (len(common_entities) / len(set(gen_entities)))
    
    nli_result = nli_model(f"premise: {reference_text} hypothesis: {generated_text}")
    entailment_score = nli_result[0]['score'] if nli_result[0]['label'] == 'ENTAILMENT' else 0
    
    combined_score = (entity_score + (1 - entailment_score)) / 2
    return combined_score

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

def read_docx(file_path):
    from docx import Document as DocxDocument
    doc = DocxDocument(file_path)
    return "\n".join([para.text for para in doc.paragraphs])
