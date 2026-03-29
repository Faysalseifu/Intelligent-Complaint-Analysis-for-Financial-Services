from __future__ import annotations

from pathlib import Path
import shutil
from typing import cast

import chromadb
import numpy as np
import pandas as pd
from chromadb.api.types import Metadata
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "raw" / "complaint_embeddings.parquet"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "full"
VECTOR_COLLECTION = "complaints"

def _normalize_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if pd.isna(value):
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def load_precomputed_embeddings(parquet_path: Path = PARQUET_PATH) -> pd.DataFrame:
    print("Loading pre-built embeddings...")
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet file not found at {parquet_path}. Please download it and place it there."
        )

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} chunks.")

    required_columns = {"chunk_text", "embedding"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in parquet: {sorted(missing_columns)}")

    return df


def build_full_vectorstore(
    parquet_path: Path = PARQUET_PATH,
    vector_store_dir: Path = VECTOR_STORE_DIR,
    collection_name: str = VECTOR_COLLECTION,
    batch_size: int = 5000,
) -> Path:
    df = load_precomputed_embeddings(parquet_path)

    texts = df["chunk_text"].tolist()
    raw_metadatas = df[[col for col in df.columns if col not in ["chunk_text", "embedding"]]].to_dict(
        orient="records"
    )

    metadatas = cast(
        list[Metadata],
        [{str(key): _normalize_value(value) for key, value in metadata.items()} for metadata in raw_metadatas],
    )
    embeddings = [np.asarray(vector, dtype=np.float32).tolist() for vector in df["embedding"].tolist()]
    ids = [f"chunk-{idx}" for idx in range(len(texts))]

    print("Building full ChromaDB index...")
    if vector_store_dir.exists():
        shutil.rmtree(vector_store_dir)
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(vector_store_dir))
    collection = client.get_or_create_collection(name=collection_name)

    for start in tqdm(range(0, len(texts), batch_size), desc="Indexing chunks"):
        end = min(start + batch_size, len(texts))
        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )

    print(f"Full vector store saved to {vector_store_dir}")
    return vector_store_dir


def main() -> None:
    build_full_vectorstore()


if __name__ == "__main__":
    main()
