import os
import pandas as pd
import torch
import faiss
from src.config import INBOX_PATH, SENT_PATH, FAISS_INDEX_PATH

def load_processed_emails() -> pd.DataFrame:
    """
    Load inbox and sent emails, filter out rows with missing body text,
    drop duplicates by Id, and add a 'folder' column for tracking.
    
    Returns:
        Cleaned and combined DataFrame ready for FAISS search.
    """
    inbox_df = pd.read_parquet(INBOX_PATH)
    sent_df = pd.read_parquet(SENT_PATH)

    inbox_df["folder"] = "inbox"
    sent_df["folder"] = "sent"

    combined_df = pd.concat([inbox_df, sent_df])
    combined_df = combined_df.drop_duplicates("Id")

    return combined_df


def load_faiss_index(folder: str = "inbox") -> faiss.IndexFlatIP:
    """
    Load FAISS index based on the specified folder ('inbox' or 'sent').

    Args:
        folder: Which folder's index to load ('inbox' or 'sent')

    Returns:
        FAISS index
    """
    index_filename = f"{folder}_embeddings.index"
    index_path = os.path.join(FAISS_INDEX_PATH, index_filename)

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at: {index_path}")

    return faiss.read_index(index_path)


def faiss_to_device(index: faiss.Index) -> faiss.Index:
    """
    Move FAISS index to GPU if available.
    Args:
        index: FAISS index to move
    Returns:
        FAISS index on GPU if available, otherwise on CPU
    """
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, index)
    return index