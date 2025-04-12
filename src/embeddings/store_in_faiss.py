"""
Script to store email embeddings into FAISS index.
"""

import os
import torch
import pandas as pd
from tqdm import tqdm
import faiss
from src.embeddings import EmailEmbedder
from src.utils import load_email_metadata
from src.config import PROCESSED_DIR, EMBEDDINGS_DIR

tqdm.pandas()

def prepare_email_for_embedding(df: pd.DataFrame) -> list:
    """
    Combine subject and body of emails for embedding.
    Args:
        df: DataFrame containing email data
    Returns:
        List of email texts with subject and body combined
    """
    subject = df.get("ExtractedSubject").fillna("")
    body = df.get("ExtractedBodyText").fillna("")
    return (subject + "\n" + body).tolist()

def batch_embed(embedder: EmailEmbedder, emails: list, batch_size: int) -> torch.Tensor:
    """
    Batch embed email texts into embeddings.
    Args:
        embedder: EmailEmbedder instance
        emails: List of email texts
        batch_size: Size of batches for embedding
    Returns:
        Tensor of email embeddings
    """
    embeddings = []
    for i in tqdm(range(0, len(emails), batch_size), desc="Embedding emails"):
        batch = emails[i: i + batch_size]
        batch_embeddings = embedder.embed_emails(batch, batch_size)
        embeddings.append(batch_embeddings.cpu())
    return torch.cat(embeddings, dim=0)

def build_faiss_index(embeddings: torch.Tensor) -> faiss.IndexFlatIP:
    """
    Build FAISS index from embeddings.
    Args:
        embeddings: Tensor of email embeddings
    Returns:
        FAISS index
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.numpy().astype("float32"))
    return index

def main():
    print("📥 Loading processed emails...")
    df = load_email_metadata()
    print(f"Loaded {len(df)} emails.")

    print("Preparing emails for embedding...")
    texts = prepare_email_for_embedding(df)

    print("Initializing email embedder...")
    embedder = EmailEmbedder()

    batch_size = 3

    print("Generating embeddings...")
    embeddings = batch_embed(embedder, texts, batch_size)
    print(f"Generated {embeddings.shape[0]} embeddings.")

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")

    print("💾 Saving FAISS index to disk...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    index_path = os.path.join(EMBEDDINGS_DIR, "embeddings.index")
    faiss.write_index(index, index_path)
    print(f"FAISS index saved at: {index_path}")

    print("Done!")

if __name__ == "__main__":
    main()