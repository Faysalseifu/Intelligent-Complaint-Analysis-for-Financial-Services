from typing import Dict, List, Optional
from langchain.schema import Document
from src import runtime

def initialize_runtime() -> Optional[str]:
    return runtime.initialize_runtime()

# Retriever function
def retrieve_chunks(query: str, k: int = 5, filter: Optional[Dict[str, str]] = None) -> List[Document]:
    """
    Embed query and retrieve top-k chunks.
    filter: e.g., {"product_category": "Credit card or prepaid card"}
    """
    runtime_error = initialize_runtime() if runtime.vectordb is None else None
    if runtime_error:
        raise ValueError(f"Vector store not initialized: {runtime_error}")

    if runtime.vectordb is None:
        raise ValueError("Vector store not initialized.")

    search_kwargs = {"k": k}
    if filter:
        search_kwargs["filter"] = filter

    retriever = runtime.vectordb.as_retriever(search_kwargs=search_kwargs)
    docs = retriever.invoke(query) # get_relevant_documents is deprecated in newer langchain
    return docs

def generate_answer(query: str, filter: Optional[Dict[str, str]] = None):
    runtime_error = initialize_runtime() if runtime.llm is None else None
    if runtime_error:
        return {"answer": f"RAG chain not initialized: {runtime_error}", "sources": []}

    try:
        docs = retrieve_chunks(query, k=5, filter=filter)
        if runtime.prompt is None or runtime.llm is None:
            return {"answer": "RAG chain not initialized: Prompt or model unavailable.", "sources": []}

        context = "\n\n".join(doc.page_content for doc in docs)
        full_prompt = runtime.prompt.format(context=context, question=query)
        answer = runtime.llm.invoke(full_prompt)

        return {
            "answer": answer,
            "sources": docs,
        }
    except Exception as e:
        return {"answer": f"Error generating answer: {e}", "sources": []}
