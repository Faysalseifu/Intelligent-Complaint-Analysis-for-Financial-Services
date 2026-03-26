import pandas as pd
import numpy as np
from typing import cast
from pathlib import Path
from tqdm import tqdm
import chromadb
from chromadb.api.types import Metadata
import shutil

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "raw" / "complaint_embeddings.parquet"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "full"
VECTOR_COLLECTION = "complaints"

# Load parquet
print("Loading pre-built embeddings...")
if not PARQUET_PATH.exists():
    raise FileNotFoundError(f"Parquet file not found at {PARQUET_PATH}. Please download it and place it there.")

df = pd.read_parquet(PARQUET_PATH)
print(f"Loaded {len(df)} chunks.")

required_columns = {"chunk_text", "embedding"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing required columns in parquet: {sorted(missing_columns)}")

# Prepare texts and metadatas
texts = df['chunk_text'].tolist()  # Assuming column name; adjust if different
# Metadata: exclude embedding and text columns
raw_metadatas = df[[col for col in df.columns if col not in ['chunk_text', 'embedding']]].to_dict(orient='records')


def _normalize_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if pd.isna(value):
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


metadatas = cast(list[Metadata], [
    {str(key): _normalize_value(value) for key, value in metadata.items()}
    for metadata in raw_metadatas
])
embeddings = [np.asarray(vector, dtype=np.float32).tolist() for vector in df["embedding"].tolist()]
ids = [f"chunk-{idx}" for idx in range(len(texts))]

# Build ChromaDB
print("Building full ChromaDB index...")
# Ensure directory is empty or handle overwrite
if VECTOR_STORE_DIR.exists():
    shutil.rmtree(VECTOR_STORE_DIR)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
collection = client.get_or_create_collection(name=VECTOR_COLLECTION)

batch_size = 5000
for start in tqdm(range(0, len(texts), batch_size), desc="Indexing chunks"):
    end = min(start + batch_size, len(texts))
    collection.upsert(
        ids=ids[start:end],
        documents=texts[start:end],
        metadatas=metadatas[start:end],
        embeddings=embeddings[start:end],
    )

print(f"Full vector store saved to {VECTOR_STORE_DIR}")
