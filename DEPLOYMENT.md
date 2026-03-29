# Deployment Guide (Hugging Face Spaces)

This project is configured for a Gradio Space deployment.

## 1) Required Files

Ensure these files are present in the repository:

- `app.py`
- `requirements.txt`
- `runtime.txt`
- `src/runtime.py`

Validate quickly:

```bash
python -m src.preflight --mode ci
```

## 2) Runtime Configuration

- Python version is pinned in `runtime.txt` (`python-3.11`)
- App launch uses:
  - `SERVER_NAME` (default `0.0.0.0`)
  - `PORT` (fallback `SERVER_PORT`, default `7860`)

## 3) Secrets and Variables in Space

In Space **Settings → Variables and secrets**:

- Required secret:
  - `HUGGINGFACEHUB_API_TOKEN`
- Optional variable:
  - `AUTO_BUILD_VECTORSTORE=1`

Use `AUTO_BUILD_VECTORSTORE=1` only when `data/raw/complaint_embeddings.parquet` exists in the Space storage.

## 4) Vector Store Strategy

Choose one:

1. Commit/persist `vector_store/full` artifacts into Space storage
2. Provide parquet input and let runtime build automatically (`AUTO_BUILD_VECTORSTORE=1`)

If using strategy (2), run a one-time check:

```bash
python -m src.preflight --mode build
```

## 5) Post-Deploy Verification

After Space build finishes:

1. Open app UI successfully
2. Ask one sample complaint question
3. Confirm source snippets appear
4. Confirm no startup warning is shown

## 6) Quick Troubleshooting

- `Missing environment variable: HUGGINGFACEHUB_API_TOKEN`
  - Add token in Space secrets and restart Space
- `Vector store directory not found` / `is empty`
  - Upload vector store artifacts or enable auto-build with parquet file
- Slow first startup
  - Expected when auto-building vector store from parquet
