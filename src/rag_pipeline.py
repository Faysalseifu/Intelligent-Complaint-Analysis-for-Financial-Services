import os
from typing import Dict, List, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.schema import Document
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "full"
VECTOR_COLLECTION = "complaints"

vectordb: Optional[Chroma] = None
llm: Optional[HuggingFaceHub] = None
rag_chain: Optional[RetrievalQA] = None

def initialize_runtime() -> Optional[str]:
    global vectordb, llm, rag_chain

    if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
        return "HUGGINGFACEHUB_API_TOKEN not found in environment."

    if not VECTOR_STORE_DIR.exists() or not any(VECTOR_STORE_DIR.iterdir()):
        return (
            f"Vector store not found at {VECTOR_STORE_DIR}. "
            "Run src/task3_build_full_vectorstore.py first."
        )

    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma(
            collection_name=VECTOR_COLLECTION,
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=embeddings,
        )
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            model_kwargs={"temperature": 0.3, "max_new_tokens": 512},
        )
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )
        return None
    except Exception as exc:
        vectordb = None
        llm = None
        rag_chain = None
        return str(exc)

# Retriever function
def retrieve_chunks(query: str, k: int = 5, filter: Optional[Dict[str, str]] = None) -> List[Document]:
    """
    Embed query and retrieve top-k chunks.
    filter: e.g., {"product_category": "Credit card or prepaid card"}
    """
    runtime_error = initialize_runtime() if vectordb is None else None
    if runtime_error:
        raise ValueError(f"Vector store not initialized: {runtime_error}")
        
    retriever = vectordb.as_retriever(search_kwargs={"k": k, "filter": filter})
    docs = retriever.invoke(query) # get_relevant_documents is deprecated in newer langchain
    return docs

# Prompt template (customize as needed)
prompt_template = """
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. 
Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.
Be concise, insightful, and evidence-based. Summarize common themes if multiple chunks match.

Context: {context}

Question: {question}

Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def generate_answer(query: str, filter: Optional[Dict[str, str]] = None):
    runtime_error = initialize_runtime() if rag_chain is None else None
    if runtime_error:
        return {"answer": f"RAG chain not initialized: {runtime_error}", "sources": []}
        
    # Langchain's RetrievalQA expects 'query' key for input usually, or just the string args depending on version
    # 'invoke' is preferred in newer versions, but 'call' or dict use works in older
    try:
        # Handling filter is tricky with standard RetrievalQA as it doesn't easily accept dynamic retriever kwargs per call
        # We might need to construct a new retriever or update the existing one if filter is provided.
        # But for simplicity, we'll assume the basic chain usage or reconstruct if filter is critical.
        
        # NOTE: Standard RetrievalQA doesn't separate search_kwargs per call easily. 
        # A workaround is to access the retriever:
        if filter:
            rag_chain.retriever.search_kwargs['filter'] = filter
        else:
            if 'filter' in rag_chain.retriever.search_kwargs:
                del rag_chain.retriever.search_kwargs['filter']
                
        result = rag_chain.invoke({"query": query})
        return {
            "answer": result["result"],
            "sources": result.get("source_documents", [])
        }
    except Exception as e:
        return {"answer": f"Error generating answer: {e}", "sources": []}
