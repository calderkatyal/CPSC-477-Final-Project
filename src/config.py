import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "embeddings")

INBOX_PATH = os.path.join(PROCESSED_DIR, "Inbox.parquet")
SENT_PATH = os.path.join(PROCESSED_DIR, "Sent.parquet")
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "embeddings")

DB_URL = "postgresql://postgres:password@localhost:5432/emails_db"
