"""
Script to store email embeddings into FAISS index.
"""

# TODO: Modify to store Inbox and Sent email embeddings separately.

import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
import faiss
from src.embeddings.embeddings import EmailEmbedder
from src.utils import load_processed_emails
from src.config import PROCESSED_DIR, EMBEDDINGS_DIR, INBOX_PATH, SENT_PATH

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

def main(args):
    print("📥 Loading processed emails...")

    df = load_processed_emails()
    inbox_df = df[df["folder"] == "inbox"]
    sent_df = df[df["folder"] == "sent"]

    print(f"Inbox: {len(inbox_df)} emails | Sent: {len(sent_df)} emails")

    print("Initializing email embedder...")
    embedder = EmailEmbedder(seed=args.seed)
    batch_size = args.batch_size

    for label, df in [("inbox", inbox_df), ("sent", sent_df)]:
        if df.empty:
            print(f"⚠️ No emails in {label}, skipping.")
            continue

        print(f"\n📤 Processing {label.capitalize()} emails...")
        texts = prepare_email_for_embedding(df)

        print("Generating embeddings...")
        embeddings = batch_embed(embedder, texts, batch_size)
        print(f"Generated {embeddings.shape[0]} embeddings for {label}.")

        print("Building FAISS index...")
        index = build_faiss_index(embeddings)
        print(f"FAISS index built with {index.ntotal} vectors.")

        print("💾 Saving FAISS index to disk...")
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        index_path = os.path.join(EMBEDDINGS_DIR, f"{label}_embeddings.index")
        faiss.write_index(index, index_path)
        print(f"FAISS index saved at: {index_path}")

    print("\nDone!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Store email embeddings in FAISS index.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for embedding emails.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)