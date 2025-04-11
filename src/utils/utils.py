import os
import pandas as pd


def load_email_metadata() -> pd.DataFrame:
    """
    Load raw and preprocessed email data.

    Returns:
         Combined DataFrame of inbox and sent emails with duplicates removed
    """
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
    inbox_path = os.path.join(PROCESSED_DIR, "Inbox.parquet")
    sent_path = os.path.join(PROCESSED_DIR, "Sent.parquet")

    inbox_df = pd.read_parquet(inbox_path)
    sent_df = pd.read_parquet(sent_path)

    inbox_df["folder"] = "inbox"
    sent_df["folder"] = "sent"

    emails_df = pd.concat([inbox_df, sent_df]).drop_duplicates("Id")
    return emails_df