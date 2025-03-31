import pandas as pd
import re
import os
from typing import Tuple
from dataloader import load, save
from datetime import datetime
import email.utils
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
tqdm.pandas()

# Compile regex patterns
PUNCTUATION_REGEX = re.compile(r'[^\w\s]')
SUBJECT_CLEAN_REGEX = re.compile(r'^(re|fwd):\s*', flags=re.IGNORECASE)

def clean_text(texts: pd.Series, is_body: bool) -> pd.Series:
    texts = texts.fillna("").astype(str)
    texts = texts.str.lower().str.replace(PUNCTUATION_REGEX, " ", regex=True)
    texts = texts.str.split().apply(lambda words: ' '.join(words))
    if not is_body:
        texts = texts.str.replace(SUBJECT_CLEAN_REGEX, "", regex=True).str.strip()
    return texts

def preprocess_hillary_emails(
    emails_path: str,
    receivers_path: str,
    aliases_path: str,
    persons_path: str,
    output_path: str,
    workers: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    logger.info("Loading Hillary Clinton dataset...")
    emails = load(emails_path)
    receivers = load(receivers_path)
    aliases = load(aliases_path)
    persons = load(persons_path)

    logger.info("Merging aliases with persons...")
    alias_map = aliases.merge(persons, left_on='PersonId', right_on='Id', suffixes=('', '_person'))
    alias_map = alias_map[['Alias', 'Name']]

    # Build alias -> person name dict
    alias_to_name = dict(zip(alias_map['Alias'].str.lower(), alias_map['Name']))

    # Add Sender Name
    emails['SenderName'] = emails['ExtractedFrom'].str.lower().map(alias_to_name)

    # Add receiver names
    receivers = receivers.merge(persons, left_on='PersonId', right_on='Id')
    email_to_recipients = receivers.groupby('EmailId')['Name'].apply(list).to_dict()

    emails['ReceiverNames'] = emails['Id'].map(email_to_recipients)

    logger.info("Cleaning subject and body...")
    emails['subject'] = clean_text(emails['ExtractedSubject'], is_body=False)
    emails['body'] = clean_text(emails['ExtractedBodyText'], is_body=True)

    # Parse date
    def parse_date(date_str):
        try:
            return pd.to_datetime(date_str)
        except Exception:
            return pd.NaT

    emails['date'] = emails['ExtractedDateSent'].apply(parse_date)

    logger.info("Splitting into Inbox and Sent folders...")

    # Define Hillary as the main person of interest
    hillary_names = persons[persons['Name'].str.contains("hillary", case=False, na=False)]['Name'].tolist()
    hillary_name = hillary_names[0] if hillary_names else "Hillary Clinton"

    # INBOX = any email where Hillary is in the ReceiverNames
    inbox_df = emails[emails['ReceiverNames'].apply(lambda x: hillary_name in x if isinstance(x, list) else False)]

    # SENT = any email where Hillary is the Sender
    sent_df = emails[emails['SenderName'] == hillary_name]

    logger.info(f"Inbox: {len(inbox_df)} emails | Sent: {len(sent_df)} emails")

    logger.info("Saving processed data...")
    save({'inbox': inbox_df, 'sent': sent_df}, output_path)

    return inbox_df, sent_df


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    RAW_DIR = os.path.join(BASE_DIR, "raw")
    PROCESSED_PATH = os.path.join(BASE_DIR, "processed", "emails_hillary.parquet")

    emails_path = os.path.join(RAW_DIR, "emails.csv")
    receivers_path = os.path.join(RAW_DIR, "emailreceivers.csv")
    aliases_path = os.path.join(RAW_DIR, "aliases.csv")
    persons_path = os.path.join(RAW_DIR, "persons.csv")

    workers = cpu_count() * 2
    inbox_df, sent_df = preprocess_hillary_emails(
        emails_path,
        receivers_path,
        aliases_path,
        persons_path,
        PROCESSED_PATH,
        workers=workers
    )

    print(f"\n📥 Inbox emails: {len(inbox_df)}")
    print(f"📤 Sent emails:  {len(sent_df)}")
