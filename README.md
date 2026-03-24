# MUM Chatbot Assistant

A Retrieval-Augmented Generation (RAG) chatbot that enables natural language querying of Middlesex University Mauritius policy documents. .

## Overview

The Chatbot allows students and staff to ask questions about university policies in plain English and receive accurate, source-grounded answers. Instead of manually searching through lengthy PDF and DOCX documents, users can simply type their question and get a relevant response with citations.

## Tech Stack

- **Language:** Python
- **LLM:** Google Gemini 2.5 Flash (generation)
- **Embeddings:** gemini-embedding-001
- **Vector Store:** ChromaDB
- **UI:** Streamlit

## Project Structure

```
├── app.py              # Streamlit web interface
├── query.py            # RAG query engine and response generation
├── ingest.py           # Document ingestion and chunking pipeline
├── maintain.py         # Maintenance utilities for the vector store
├── evaluate.py         # Evaluation suite with retrieval and generation metrics
├── app_icon.py         # Application icon configuration
├── style.css           # Custom Streamlit UI styling
├── test_dataset.json   # 30-query evaluation dataset with ground truth
├── requirements.txt    # Python dependencies
└── chatbot_icon.png    # Chatbot avatar
```

## Features

- Natural language querying of MUM policy documents
- Source-grounded responses with document citations
- Follow-up question suggestions
- Chat session history
- Feedback collection system
- Incremental document re-indexing with MD5 hash-based deduplication
- Language translation and providing responses in French 

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/imben03/RAG_Chatbot_Project.git
cd RAG_Chatbot_Project
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 5. Prepare documents

Place your MUM policy documents (PDF and DOCX) in a `docs/` folder in the project root.

### 6. Ingest documents

```bash
python ingest.py
```

### 7. Run the application

```bash
streamlit run app.py
```

## Evaluation

The project includes a comprehensive evaluation suite (`evaluate.py`) that measures:

- **Retrieval metrics:** Precision@k, Recall@k, Context Precision
- **Generation metrics:** Faithfulness, Answer Relevancy (scored by Gemini)
- **Performance:** Latency and P95 response time
- **Robustness:** Variance across paraphrased queries
- **Baseline comparison:** RAG vs no-retrieval generation

Run the evaluation:

```bash
python evaluate.py
```

## Author

Developed as a  capstone project at Middlesex University Mauritius.
