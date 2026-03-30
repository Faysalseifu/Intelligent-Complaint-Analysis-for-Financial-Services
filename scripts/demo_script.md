# Demo Script (5-7 minutes)

## 1) Opening (30s)
- Introduce goal: grounded answers over CFPB-style complaints.
- Mention architecture: precomputed embeddings -> Chroma retrieval -> HF-hosted LLM answer generation.

## 2) Startup Validation (45s)
- Run:
  - `python scripts/smoke_test.py --skip-app-check`
- Explain what this confirms:
  - required files present
  - tests pass in current repo state

## 3) App Launch (30s)
- Run:
  - `python app.py`
- Open local URL and point out:
  - chat panel
  - product filter
  - source panel

## 4) Happy Path Query (2 min)
- Ask one concrete question, e.g.:
  - "What are common issues reported for credit cards?"
- Narrate expected behavior:
  - streamed answer
  - concise summary
  - source excerpts with metadata

## 5) Filtered Query (1 min)
- Set product filter to one category (e.g., Mortgage).
- Ask a similar query.
- Show how retrieval context changes with metadata filter.

## 6) Failure Path (1 min)
- Describe one controlled failure scenario:
  - missing token or missing vector store
- Show app/runtime error message is explicit and actionable.

## 7) Close (30s)
- Summarize scope:
  - demo-ready reliability
  - deployable on Hugging Face Spaces
  - transparent grounding via source snippets
- Mention known limitations and next improvements from README.
