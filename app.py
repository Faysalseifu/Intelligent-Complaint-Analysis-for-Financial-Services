"""Gradio chat UI for the RAG system with streaming answers and source display."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gradio as gr
from langchain.schema import Document
from src import runtime

def initialize_runtime() -> Optional[str]:
    return runtime.initialize_runtime()


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
    if runtime.vectordb is None:
        raise RuntimeError("Vector store is not initialized.")

    filter_dict: Optional[Dict[str, str]] = None
    if product_filter and product_filter != "All":
        filter_dict = {"product_category": product_filter}
    return runtime.vectordb.similarity_search(query, k=5, filter=filter_dict)


def _build_prompt(query: str, docs: List[Document]) -> str:
    if runtime.prompt is None:
        raise RuntimeError("Prompt is not initialized.")

    context = "\n\n".join(doc.page_content for doc in docs)
    return runtime.prompt.format(context=context, question=query)


def chat_generator(message: str, history: List[Tuple[str, str]], product_filter: str):
    history = history or []
    history.append((message, ""))
    startup_error = initialize_runtime()
    if startup_error:
        history[-1] = (message, startup_error)
        yield history, "**Runtime Error**\n\nCheck token and vector store setup.", ""
        return

    docs = _retrieve_docs(message, product_filter)
    full_prompt = _build_prompt(message, docs)
    answer = ""
    if runtime.llm is None:
        history[-1] = (message, "Language model is not initialized.")
        yield history, _format_sources(docs), ""
        return

    for chunk in runtime.llm.stream(full_prompt):
        answer += chunk
        history[-1] = (message, answer)
        yield history, _format_sources(docs), ""
    yield history, _format_sources(docs), ""


def clear_chat():
    return [], "", ""


def build_ui():
    startup_error = initialize_runtime()
    with gr.Blocks(title="CrediTrust Complaint Insights", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # CrediTrust Complaint Insights Chatbot
            Ask about CFPB complaints. Answers are grounded in retrieved chunks; sources shown below.
            """
        )
        if startup_error:
            gr.Markdown(f"**Startup warning:** {startup_error}")

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
