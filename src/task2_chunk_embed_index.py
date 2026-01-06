"""Task 2: Chunk complaints, embed with MiniLM, and build a Chroma vector store.

Run with:
    python src/task2_chunk_embed_index.py
"""
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILTERED_CSV = PROJECT_ROOT / "data" / "processed" / "filtered_complaints.csv"
SAMPLED_CSV = PROJECT_ROOT / "data" / "processed" / "sampled_complaints.csv"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "sample"


def load_filtered_dataframe() -> pd.DataFrame:
    """Load the filtered complaints CSV produced in Task 1."""
    if not FILTERED_CSV.exists():
        raise FileNotFoundError(f"Filtered complaints not found at: {FILTERED_CSV}")

    df = pd.read_csv(FILTERED_CSV)
    required_cols = [
        "clean_narrative",
        "Product",
        "Issue",
        "Sub-issue",
        "Company",
        "State",
        "Date received",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in filtered data: {missing}")

    # Ensure unique identifier
    if "Complaint ID" in df.columns:
        df["complaint_id"] = df["Complaint ID"]
    else:
        df["complaint_id"] = df.index.astype(str)

    # Drop rows without narratives or products to keep stratification stable
    df = df.dropna(subset=["clean_narrative", "Product"]).reset_index(drop=True)
    return df


def stratified_sample(df: pd.DataFrame, target_size: int = 12_000) -> pd.DataFrame:
    """Create a stratified sample by Product, then trim/expand toward the target size."""
    if df.empty:
        raise ValueError("Filtered dataframe is empty; nothing to sample.")

    # Initial ~2% take rate to land near the target for typical CFPB volumes
    sample_df, _ = train_test_split(
        df,
        test_size=0.98,
        stratify=df["Product"],
        random_state=42,
    )

    if len(sample_df) > target_size:
        sample_df = sample_df.sample(n=target_size, random_state=42)
    elif len(sample_df) < target_size:
        print(f"Warning: sample size {len(sample_df)} < target {target_size}. Adjust test_size if needed.")

    return sample_df.reset_index(drop=True)


def chunk_narratives(df: pd.DataFrame) -> Tuple[List[str], List[dict]]:
    """Split complaint narratives into overlapping character chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )

    texts: List[str] = []
    metadatas: List[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        narrative = row["clean_narrative"]
        if not isinstance(narrative, str) or not narrative.strip():
            continue

        chunks = splitter.split_text(narrative)
        for chunk_idx, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append(
                {
                    "complaint_id": row["complaint_id"],
                    "product_category": row["Product"],
                    "issue": row.get("Issue", ""),
                    "sub_issue": row.get("Sub-issue", ""),
                    "company": row["Company"],
                    "state": row["State"],
                    "date_received": row["Date received"],
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "source": "cfpb_complaint",
                }
            )

    return texts, metadatas


def build_vector_store(texts: List[str], metadatas: List[dict]) -> Chroma:
    """Create and persist a Chroma vector store with MiniLM embeddings."""
    if not texts:
        raise ValueError("No chunks were produced; cannot build vector store.")

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=str(VECTOR_STORE_DIR),
    )
    vectordb.persist()
    return vectordb


def test_retrieval(vectordb: Chroma) -> None:
    """Run a quick similarity search sanity check."""
    query = "billing disputes on credit cards"
    results = vectordb.similarity_search(query, k=5)

    print("\nTop 5 retrieved chunks:")
    for idx, doc in enumerate(results, start=1):
        preview = doc.page_content[:300].replace("\n", " ")
        print(f"\n--- Result {idx} ---")
        print(f"Text: {preview}...")
        print(
            "Metadata: Product={product}, Complaint ID={cid}".format(
                product=doc.metadata.get("product_category"),
                cid=doc.metadata.get("complaint_id"),
            )
        )


def main() -> None:
    print("Loading filtered complaints...")
    df = load_filtered_dataframe()

    print("Creating stratified sample (~12k target)...")
    sample_df = stratified_sample(df)
    print(f"Sample size: {len(sample_df)}")
    print(sample_df["Product"].value_counts(normalize=True))

    print(f"Saving sample to {SAMPLED_CSV}")
    SAMPLED_CSV.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(SAMPLED_CSV, index=False)

    print("Generating chunks...")
    texts, metadatas = chunk_narratives(sample_df)
    print(f"Generated {len(texts)} chunks from {len(sample_df)} complaints.")

    print("Building ChromaDB index (this may take 10-20 minutes)...")
    vectordb = build_vector_store(texts, metadatas)
    print(f"Vector store saved to {VECTOR_STORE_DIR}")

    print("\nTesting retrieval...")
    test_retrieval(vectordb)


if __name__ == "__main__":
    main()
