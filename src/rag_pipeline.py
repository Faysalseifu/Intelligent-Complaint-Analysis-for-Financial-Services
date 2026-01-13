import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "full"

# Set HF token
# IMPORTANT: Replace with your actual token or set it in your environment variables
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    # Placeholder warning
    print("Warning: HUGGINGFACEHUB_API_TOKEN not found in environment. Please set it.")
    # os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_your_token_here"

# Load vector store
print("Loading vector store...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Check if vector store exists
if not VECTOR_STORE_DIR.exists():
    print(f"Vector store not found at {VECTOR_STORE_DIR}. Please run src/task3_build_full_vectorstore.py first.")
    vectordb = None
else:
    vectordb = Chroma(persist_directory=str(VECTOR_STORE_DIR), embedding_function=embeddings)

# Retriever function
def retrieve_chunks(query, k=5, filter=None):
    """
    Embed query and retrieve top-k chunks.
    filter: e.g., {"product_category": "Credit card or prepaid card"}
    """
    if vectordb is None:
        raise ValueError("Vector store not initialized.")
        
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

# LLM setup
try:
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs={"temperature": 0.3, "max_length": 512}
    )
except Exception as e:
    print(f"Error initializing LLM: {e}")
    llm = None

# Full RAG chain
if llm and vectordb:
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Stuff context into prompt
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT}
    )
else:
    rag_chain = None

# Generator function
def generate_answer(query, filter=None):
    if rag_chain is None:
        return {"answer": "RAG chain not initialized (check LLM or VectorDB)", "sources": []}
        
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
            "sources": result["source_documents"]  # List of Document objects with metadata
        }
    except Exception as e:
        return {"answer": f"Error generating answer: {e}", "sources": []}
