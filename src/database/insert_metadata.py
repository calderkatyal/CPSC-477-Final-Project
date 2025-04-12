import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from tqdm import tqdm
from src.config import DB_URL, INBOX_PATH, SENT_PATH

tqdm.pandas()

inbox = pd.read_parquet(INBOX_PATH)
sent = pd.read_parquet(SENT_PATH)

# Add folder label
inbox["folder"] = "inbox"
sent["folder"] = "sent"

# Combine and deduplicate
emails = pd.concat([inbox, sent]).drop_duplicates("Id")

# Rename columns to match schema
emails_to_sql = emails.rename(columns={
    "Id": "id",
    "ExtractedSubject": "subject",
    "ExtractedBodyText": "body",
    "ExtractedFrom": "sender",
    "ExtractedTo": "recipients",
    "ExtractedCc": "cc",
    "ExtractedDateSent": "date_sent"
})

# Convert recipients and cc to string
emails_to_sql['recipients'] = emails_to_sql['recipients'].progress_apply(
    lambda x: ', '.join(x) if isinstance(x, (list, np.ndarray)) else x
)

emails_to_sql['cc'] = emails_to_sql['cc'].progress_apply(
    lambda x: ', '.join(x) if isinstance(x, (list, np.ndarray)) else x
)

# Connect to PostgreSQL
engine = create_engine(DB_URL)

# Save to PostgreSQL
print(f"📥 Inserting {len(emails_to_sql)} rows into PostgreSQL...")
emails_to_sql.to_sql("emails", engine, index=False, if_exists="replace")

print("Metadata stored in PostgreSQL!")
