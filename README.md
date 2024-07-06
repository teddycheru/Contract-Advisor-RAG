# Contract Q&A System

## Overview

The Contract Q&A System is an application designed to help users query contract documents and evaluate the answers using various metrics. This system uses advanced natural language processing techniques to perform similarity searches and provide relevant answers based on the content of the uploaded contract files.

## Features

- Upload contract files in `.docx` format.
- Chunk contracts for efficient processing.
- Perform similarity searches to find relevant answers to user queries.
- Evaluate the generated answers using BLEU score and hallucination metrics.
- Streamlit-based user interface for easy interaction.

## File Structure
```
├── data
│   ├── contracts
│   │   └── your_contract_files.docx
│   ├── qna
│       └── your_evaluation_data.docx
│       └── your_evaluation_data.docx
│
├── notebooks
│   └── simple_qna.ipynb
│
├── screencaptures
│   ├── screencast
│   │   └──Screencast.webm
│   ├── screenshots
│       ├──1.png
│       ├──2.png
│       └──3.png
│
├── src
│   ├── generator.py
│   ├── query.py
│   ├── retriever.py
│   └── evaluator.py
│
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```
  

## Tech Stack

- **Programming Language**: Python
- **Libraries**:
  - `streamlit` for the user interface
  - `python-docx` for reading `.docx` files
  - `spacy` for natural language processing
  - `transformers` for natural language inference
  - `sacrebleu` for BLEU score calculation
  - `dotenv` for environment variable management

## Installation

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/teddycheru/Contract-Advisor-RAG.git
   cd Contract-Advisor-RAG

2. **Create and Activate a Virtual Environment**:
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

3. **Install the Dependencies**:
   ```sh
   pip install -r requirements.txt
   
4. **Set Up Environment Variables**:
   
   Create a `.env` file in the project root directory.
   Add your `OpenAI API` key to the `.env` file:
   ```sh
   OPENAI_API_KEY=your_openai_api_key_here

## Usage
1. **Run the Streamlit App**:
   ```sh
   streamlit run query.py
   
2. **Upload Contracts**:
- Navigate to the Streamlit app in your browser.
- Use the sidebar to upload .docx contract files.

3. **Query Contracts**:
- Enter your query in the text input field.
- View the generated answer and its evaluation metrics.

4. **Bulk Evaluation**:
- Use the sidebar button to run a bulk evaluation on predefined queries and references from the evaluation data.


