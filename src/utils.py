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



def load_faiss_index() -> faiss.IndexFlatIP:
    """
    Loads the FAISS index.

    Returns:
        FAISS index
    """
    return faiss.read_index(FAISS_INDEX_PATH)

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