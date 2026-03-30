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

Launch configuration can be controlled with environment variables:

- `SERVER_NAME` (default: `0.0.0.0`)
- `PORT` or `SERVER_PORT` (default: `7860`)
- `AUTO_BUILD_VECTORSTORE` (`1` to auto-build from parquet when vector store is missing)

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

Detailed checklist: see `DEPLOYMENT.md`.

## Current Release Scope

This release targets **demo-ready** quality:

- Stable startup and runtime checks
- Basic tests for readiness/error paths
- Deployable Gradio app on Hugging Face Spaces

For production hardening, add stronger evaluation, observability, and stricter security/compliance controls.

## Day 5 Launch-Readiness Checklist

Use this checklist before demo day:

1. Fresh install works:

```bash
pip install -r requirements.txt
python -m src.preflight --mode ci
```

2. Runtime readiness passes:

```bash
python -m src.preflight --mode app
```

3. Launch app and validate one happy-path query:

```bash
python app.py
```

4. Validate one expected failure path (for example, start without token and confirm clear error).

5. Run tests:

```bash
pytest -q
```

## Smoke Test Script

Run automated launch smoke checks:

```bash
python scripts/smoke_test.py
```

Optional flags:

- `--skip-app-check` to skip token/vector-store runtime checks
- `--strict-runtime` to fail hard if runtime checks fail

## Demo Script

Use the presenter runbook in `scripts/demo_script.md` for a consistent walkthrough.

## Known Limitations

- Retrieval uses semantic similarity only (no hybrid BM25 + dense reranking).
- Hosted LLM latency depends on Hugging Face Inference API availability and rate limits.
- Temporal trend questions are limited to what appears in retrieved chunks.
- Fresh data ingestion requires rebuilding the persisted vector store.

## Next Improvements

- Add reranking and optional hybrid retrieval.
- Add structured evaluation metrics and regression checks.
- Add lightweight observability (request timings, retrieval diagnostics).
- Add incremental index update workflow for new complaints.
