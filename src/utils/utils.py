import os
import pandas as pd
import faiss
from config import INBOX_PATH, SENT_PATH

def load_email_metadata() -> pd.DataFrame:
    """
    Load raw and preprocessed email data.

    Returns:
         Combined DataFrame of inbox and sent emails with duplicates removed
    """
    inbox_df = pd.read_parquet(INBOX_PATH)
    sent_df = pd.read_parquet(SENT_PATH)

    inbox_df["folder"] = "inbox"
    sent_df["folder"] = "sent"

    emails_df = pd.concat([inbox_df, sent_df])
    return emails_df

def load_faiss_index() -> faiss.IndexFlatIP:
    """
    Loads the FAISS index.

    Returns:
        FAISS index
    """
    index_path = os.path.join(EMBEDDINGS_DIR, "embeddings.index")

