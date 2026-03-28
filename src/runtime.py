from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Chroma

PROJECT_ROOT = Path(__file__).parent.parent
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "full"
VECTOR_COLLECTION = "complaints"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO = "mistralai/Mistral-7B-Instruct-v0.2"

PROMPT_TEMPLATE = (
    "You are a helpful financial analyst for CrediTrust. "
    "Answer the question based ONLY on the provided complaint excerpts. "
    "If the context lacks the answer, say you do not have enough information. "
    "Be concise, highlight recurring themes, and avoid speculation.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

qa_chain: Optional[RetrievalQA] = None
llm: Optional[HuggingFaceHub] = None
vectordb: Optional[Chroma] = None
prompt: Optional[PromptTemplate] = None
runtime_error: Optional[str] = None
_runtime_initialized = False


def _require_token() -> str:
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError("Set HUGGINGFACEHUB_API_TOKEN before launching the app.")
    return token


def _load_vectorstore() -> Chroma:
    if not VECTOR_STORE_DIR.exists() or not any(VECTOR_STORE_DIR.iterdir()):
        raise FileNotFoundError(
            f"Vector store not found at {VECTOR_STORE_DIR}. "
            "Build it first with: python src/task3_build_full_vectorstore.py"
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name=VECTOR_COLLECTION,
        persist_directory=str(VECTOR_STORE_DIR),
        embedding_function=embeddings,
    )


def _build_chain(vectordb: Chroma) -> Tuple[RetrievalQA, HuggingFaceHub, PromptTemplate]:
    _require_token()

    model = HuggingFaceHub(
        repo_id=LLM_REPO,
        model_kwargs={"temperature": 0.35, "max_new_tokens": 512},
    )

    prompt_template = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )
    return chain, model, prompt_template


def initialize_runtime(force_reinitialize: bool = False) -> Optional[str]:
    global qa_chain, llm, vectordb, prompt, runtime_error, _runtime_initialized

    if force_reinitialize:
        _runtime_initialized = False

    if _runtime_initialized:
        return runtime_error

    try:
        vectordb = _load_vectorstore()
        qa_chain, llm, prompt = _build_chain(vectordb)
        runtime_error = None
    except Exception as exc:
        qa_chain = None
        llm = None
        vectordb = None
        prompt = None
        runtime_error = (
            "Runtime not ready: "
            f"{exc}. Ensure HUGGINGFACEHUB_API_TOKEN is set and the vector store exists."
        )

    _runtime_initialized = True
    return runtime_error


def reset_runtime_state() -> None:
    global qa_chain, llm, vectordb, prompt, runtime_error, _runtime_initialized
    qa_chain = None
    llm = None
    vectordb = None
    prompt = None
    runtime_error = None
    _runtime_initialized = False
