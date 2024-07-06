import streamlit as st
from retriever import load_contracts_from_uploaded_files
from generator import chunk_contracts, create_docstore
from evaluator import load_evaluation_data, bleu, calculate_hallucination_score, calculate_additional_metrics
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit app
st.set_page_config(
    page_title="Contract Q&A System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }
    .stTextInput input {
        border: 2px solid #ccc;
        border-radius: 4px;
        padding: 8px;
        width: 100%;
    }
    .stTextArea textarea {
        border: 2px solid #ccc;
        border-radius: 4px;
        padding: 8px;
        width: 100%;
    }
    .stSelectbox div {
        border: 2px solid #ccc;
        border-radius: 4px;
        padding: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 Contract Q&A System")
st.markdown("### Upload contract files and ask questions about their content.")

# Sidebar for file upload and evaluation
st.sidebar.header("Upload Contracts")
uploaded_files = st.sidebar.file_uploader("Upload contract files", type=["docx"], accept_multiple_files=True)

# Load and process contracts
if uploaded_files:
    contracts = load_contracts_from_uploaded_files(uploaded_files)
    documents = chunk_contracts(contracts)
    docstore = create_docstore(documents, openai_api_key)

    # Load evaluation data
    EVAL_FILE = "../data/qna/Robinson Q&A.docx"
    evaluation_data = load_evaluation_data(EVAL_FILE)

    def query_function(query):
        results = docstore.similarity_search(query, k=5)
        return results[0].page_content if results else ""

    st.sidebar.markdown("### Evaluation Metrics")

    user_query = st.text_input("Enter your query:")

    if user_query:
        with st.spinner("Generating answer..."):
            generated_answer = query_function(user_query)

        st.write("### Generated Answer")
        st.info(generated_answer)

        reference = next((qa['answer'] for qa in evaluation_data if qa['question'] == user_query), None)
        if reference:
            bleu_score_value = bleu(generated_answer, reference)
            hallucination_score_value = calculate_hallucination_score(generated_answer, reference)
            additional_metrics = calculate_additional_metrics(generated_answer, reference)

            st.write("### Evaluation Metrics")
            st.write(f"**BLEU Score**: {bleu_score_value:.2f}")
            st.write(f"**Hallucination Score**: {hallucination_score_value:.2f}")
            if additional_metrics:
                st.write(f"**Additional Metrics**: {additional_metrics}")
        else:
            st.warning("No reference answer available for this query.")

    # Prepare queries and references from evaluation data
    queries = [qa['question'] for qa in evaluation_data]
    references = [qa['answer'] for qa in evaluation_data]

    if st.sidebar.button("Run Bulk Evaluation"):
        total_bleu_score = 0
        total_hallucination_score = 0
        num_samples = len(queries)

        st.sidebar.write("### Bulk Evaluation Results")
        with st.spinner("Running bulk evaluation..."):
            for query, reference in zip(queries, references):
                generated_answer = query_function(query)
                bleu_score_value = bleu(generated_answer, reference)
                hallucination_score_value = calculate_hallucination_score(generated_answer, reference)
                additional_metrics = calculate_additional_metrics(generated_answer, reference)

                total_bleu_score += bleu_score_value
                total_hallucination_score += hallucination_score_value

            avg_bleu_score = total_bleu_score / num_samples
            avg_hallucination_score = total_hallucination_score / num_samples

            st.sidebar.write(f"**Average BLEU Score**: {avg_bleu_score:.2f}")
            st.sidebar.write(f"**Average Hallucination Score**: {avg_hallucination_score:.2f}")
else:
    st.write("Please upload contract files to proceed.")
