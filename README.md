# Intelligent Complaint Analysis for Financial Services

RAG chatbot for analyzing CFPB-style financial complaints with source-grounded answers.

## Features

- Gradio chat interface with streaming responses
- Retrieval-augmented generation (RAG) over complaint chunks
- Product-level filtering during retrieval
- Source excerpts shown with each answer

## Project Structure

- `app.py`: Gradio app entry point
- `src/rag_pipeline.py`: retrieval + answer generation helpers
- `src/task2_chunk_embed_index.py`: chunking/embedding pipeline
- `src/task3_build_full_vectorstore.py`: builds full Chroma index from precomputed embeddings
- `data/raw/complaints.csv`: raw complaint dataset
- `vector_store/full`: persisted Chroma database

## Prerequisites

- Python 3.11 (recommended for compatibility)
- Hugging Face token with inference access

Set token:

```bash
# Linux / macOS
export HUGGINGFACEHUB_API_TOKEN=your_token_here

# Windows PowerShell
$env:HUGGINGFACEHUB_API_TOKEN="your_token_here"
```

## Installation

```bash
pip install -r requirements.txt
```

## Build the Vector Store

This app expects a full Chroma store under `vector_store/full`.

If using precomputed parquet embeddings:

1. Place file at `data/raw/complaint_embeddings.parquet`
2. Run:

```bash
python src/task3_build_full_vectorstore.py
```

## Run the App

```bash
python app.py
```

Default local URL: `http://127.0.0.1:7860`

## Tests

```bash
pytest -q
```

## Preflight Checks

Run lightweight readiness checks before deployment or runtime:

```bash
# CI-safe checks (required files)
python -m src.preflight --mode ci

# App runtime checks (token + vector store)
python -m src.preflight --mode app

# Embedding build checks (parquet input)
python -m src.preflight --mode build
```

## Deploy to Hugging Face Spaces (Gradio)

1. Create a new Space (SDK: **Gradio**)
2. Push this repository content
3. In Space Settings → Variables and secrets, add:
	- `HUGGINGFACEHUB_API_TOKEN`
4. Ensure `vector_store/full` is present in the Space storage or built during startup workflow
5. Space will launch using `app.py`
6. Keep runtime aligned with `runtime.txt` (Python 3.11)

## Current Release Scope

This release targets **demo-ready** quality:

- Stable startup and runtime checks
- Basic tests for readiness/error paths
- Deployable Gradio app on Hugging Face Spaces

For production hardening, add stronger evaluation, observability, and stricter security/compliance controls.
