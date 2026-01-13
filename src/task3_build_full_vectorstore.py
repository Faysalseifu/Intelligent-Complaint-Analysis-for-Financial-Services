import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import shutil

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "raw" / "complaint_embeddings.parquet"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "full"

# Load parquet
print("Loading pre-built embeddings...")
if not PARQUET_PATH.exists():
    raise FileNotFoundError(f"Parquet file not found at {PARQUET_PATH}. Please download it and place it there.")

df = pd.read_parquet(PARQUET_PATH)
print(f"Loaded {len(df)} chunks.")

# Prepare texts and metadatas
texts = df['chunk_text'].tolist()  # Assuming column name; adjust if different
# Metadata: exclude embedding and text columns
metadatas = df[[col for col in df.columns if col not in ['chunk_text', 'embedding']]].to_dict(orient='records')

# Embeddings are pre-computed, so we use a fake embedder for Chroma (it won't re-embed)
class PrecomputedEmbeddings:
    def embed_documents(self, texts):
        # Return pre-computed as list of lists
        # We assume the 'embedding' column contains the vector (list/array)
        return df['embedding'].tolist()
        
    def embed_query(self, text):
        # For queries, use actual model
        actual_embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return actual_embedder.embed_query(text)

embedder = PrecomputedEmbeddings()

# Build ChromaDB
print("Building full ChromaDB index...")
# Ensure directory is empty or handle overwrite
if VECTOR_STORE_DIR.exists():
    shutil.rmtree(VECTOR_STORE_DIR)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

vectordb = Chroma.from_texts(
    texts=texts,
    embedding=embedder,
    metadatas=metadatas,
    persist_directory=str(VECTOR_STORE_DIR)
)
vectordb.persist()
print(f"Full vector store saved to {VECTOR_STORE_DIR}")
