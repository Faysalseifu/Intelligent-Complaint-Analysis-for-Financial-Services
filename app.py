"""Gradio chat UI for the RAG system with streaming answers and source display."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import gradio as gr
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Chroma

PROJECT_ROOT = Path(__file__).parent
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "full"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO = "mistralai/Mistral-7B-Instruct-v0.2"


def _require_token() -> str:
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError("Set HUGGINGFACEHUB_API_TOKEN before launching the app.")
    return token


def _load_vectorstore() -> Chroma:
    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at {VECTOR_STORE_DIR}. Run src/task3_build_full_vectorstore.py first."
        )
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=str(VECTOR_STORE_DIR), embedding_function=embeddings)


def _build_chain() -> Tuple[RetrievalQA, HuggingFaceHub, Chroma, PromptTemplate]:
    _require_token()
    vectordb = _load_vectorstore()
    llm = HuggingFaceHub(
        repo_id=LLM_REPO,
        model_kwargs={"temperature": 0.35, "max_new_tokens": 512},
    )

    prompt_template = (
        "You are a helpful financial analyst for CrediTrust. "
        "Answer the question based ONLY on the provided complaint excerpts. "
        "If the context lacks the answer, say you do not have enough information. "
        "Be concise, highlight recurring themes, and avoid speculation.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain, llm, vectordb, prompt


qa_chain, llm, vectordb, prompt = _build_chain()


def _format_sources(docs: List[Document]) -> str:
    if not docs:
        return "**No sources retrieved.**"
    lines = ["**Retrieved Sources**\n"]
    for idx, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        lines.append(
            f"**Source {idx}** — Product: {meta.get('product_category', 'N/A')}; "
            f"Issue: {meta.get('issue', 'N/A')}; Complaint ID: {meta.get('complaint_id', 'N/A')}\n"
            f"Excerpt: {doc.page_content[:320]}...\n"
        )
    return "\n".join(lines)


def _retrieve_docs(query: str, product_filter: Optional[str]) -> List[Document]:
    filter_dict: Optional[Dict[str, str]] = None
    if product_filter and product_filter != "All":
        filter_dict = {"product_category": product_filter}
    return vectordb.similarity_search(query, k=5, filter=filter_dict)


def _build_prompt(query: str, docs: List[Document]) -> str:
    context = "\n\n".join(doc.page_content for doc in docs)
    return prompt.format(context=context, question=query)


def chat_generator(message: str, history: List[Tuple[str, str]], product_filter: str):
    history = history or []
    history.append((message, ""))
    docs = _retrieve_docs(message, product_filter)
    full_prompt = _build_prompt(message, docs)
    answer = ""
    for chunk in llm.stream(full_prompt):
        answer += chunk
        history[-1] = (message, answer)
        yield history, _format_sources(docs), ""
    yield history, _format_sources(docs), ""


def clear_chat():
    return [], "", ""


def build_ui():
    with gr.Blocks(title="CrediTrust Complaint Insights", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # CrediTrust Complaint Insights Chatbot
            Ask about CFPB complaints. Answers are grounded in retrieved chunks; sources shown below.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=480, label="Chat")
                msg = gr.Textbox(placeholder="Ask about complaints...", lines=3, show_label=False)
                with gr.Row():
                    ask_btn = gr.Button("Ask", variant="primary")
                    clear_btn = gr.Button("Clear")
            with gr.Column(scale=1):
                product_dropdown = gr.Dropdown(
                    choices=[
                        "All",
                        "Credit card or prepaid card",
                        "Payday loan, title loan, or personal loan",
                        "Checking or savings account",
                        "Money transfer, virtual currency, or money service",
                        "Mortgage",
                    ],
                    value="All",
                    label="Filter by Product",
                )
                sources_md = gr.Markdown("**Sources will appear here.**", label="Sources")

        ask_btn.click(
            chat_generator,
            inputs=[msg, chatbot, product_dropdown],
            outputs=[chatbot, sources_md, msg],
        )
        msg.submit(
            chat_generator,
            inputs=[msg, chatbot, product_dropdown],
            outputs=[chatbot, sources_md, msg],
        )
        clear_btn.click(clear_chat, outputs=[chatbot, sources_md, msg])

    return demo


def main() -> None:
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
