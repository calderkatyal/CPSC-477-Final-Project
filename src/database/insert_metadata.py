import pandas as pd
from sqlalchemy import create_engine
import os

# --- Load processed data ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
inbox_path = os.path.join(DATA_DIR, "Inbox.parquet")
sent_path = os.path.join(DATA_DIR, "Sent.parquet")

inbox = pd.read_parquet(inbox_path)
sent = pd.read_parquet(sent_path)

# --- Add folder label ---
inbox["folder"] = "inbox"
sent["folder"] = "sent"

# --- Combine and deduplicate ---
emails = pd.concat([inbox, sent]).drop_duplicates("Id")

# --- Prepare PostgreSQL connection ---
DB_URL = "postgresql://postgres:password@localhost:5432/emails_db"
engine = create_engine(DB_URL)

# --- Rename columns to match schema ---
emails_to_sql = emails.rename(columns={
    "Id": "id",
    "ExtractedSubject": "subject",
    "ExtractedBodyText": "body",
    "ExtractedFrom": "sender",
    "ExtractedTo": "recipients",
    "ExtractedCc": "cc",
    "ExtractedDateSent": "date_sent"
})

# --- Save to PostgreSQL ---
print(f"📥 Inserting {len(emails_to_sql)} rows into PostgreSQL...")
emails_to_sql.to_sql("emails", engine, index=False, if_exists="replace")

print("✅ Metadata stored in PostgreSQL, with folder column!")
