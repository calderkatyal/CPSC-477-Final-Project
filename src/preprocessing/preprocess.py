import pandas as pd
import re
import os
from typing import Dict
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

def parse_headers(txt: str) -> dict:
    """Parse subject, date, and body from raw message text."""
    headers = {
        "subject": "No Subject",
        "date": None,
        "body": txt.strip()
    }

    patterns = {
        "subject": r"^Subject:\s*(.*)$",
        "date": r"^Date:\s*(.*)$"
    }
    for field, pattern in patterns.items():
        match = re.search(pattern, txt, re.MULTILINE)
        if match:
            headers[field] = match.group(1)

    # Parse date
    if headers["date"]:
        try:
            date_tuple = email.utils.parsedate_tz(headers["date"])
            if date_tuple:
                headers["date"] = datetime.fromtimestamp(email.utils.mktime_tz(date_tuple))
        except Exception:
            logger.debug(f"Date parse failed: {headers['date']}")

    # Extract body
    parts = txt.split("\n\n", 1)
    if len(parts) > 1:
        headers["body"] = parts[1].strip()

    return headers

def clean_text(texts: pd.Series, is_body: bool) -> pd.Series:
    """Lowercase, remove punctuation, and normalize whitespace."""
    texts = texts.fillna("").astype(str)
    texts = texts.str.lower().str.replace(PUNCTUATION_REGEX, " ", regex=True)
    texts = texts.str.split().apply(lambda words: ' '.join([word for word in words]))

    if not is_body:
        texts = texts.str.replace(SUBJECT_CLEAN_REGEX, "", regex=True).str.strip()
    return texts

def process_row(message: str) -> dict:
    return parse_headers(message)

def parallel_preprocess(df: pd.DataFrame, workers: int = None) -> pd.DataFrame:
    workers = workers or cpu_count() * 2
    with ProcessPoolExecutor(max_workers=workers) as executor:
        parsed_list = list(tqdm(executor.map(process_row, df["message"]), total=len(df), desc="Processing emails"))

    parsed_df = pd.DataFrame(parsed_list)
    df = pd.concat([df.reset_index(drop=True), parsed_df], axis=1)

    df["subject"] = clean_text(df["subject"], is_body=False)
    df["body"] = clean_text(df["body"], is_body=True)

    return df

def organize_by_person(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group emails by top-level mailbox owner only."""
    df = df.copy()
    df['person'] = df['file'].apply(lambda x: str(x).split('/')[0])
    return dict(tuple(df.groupby('person')))

def preprocess_emails(input_path: str, output_path: str, workers: int = None) -> Dict[str, pd.DataFrame]:
    logger.info("Loading raw emails...")
    df = load(input_path)

    logger.info(f"Loaded {len(df)} emails from raw file.")

    logger.info("Parsing and cleaning emails...")
    processed_df = parallel_preprocess(df, workers=workers)

    logger.info("Organizing emails by mailbox owner...")
    person_dfs = organize_by_person(processed_df)

    logger.info("Saving cleaned dataset...")
    save(processed_df, output_path)

    logger.info(f"✅ Completed: {len(person_dfs)} people with {len(processed_df)} emails total.")
    return person_dfs

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    RAW_CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "emails.csv")
    PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "emails.parquet")

    workers = cpu_count() * 2
    person_dfs = preprocess_emails(RAW_CSV_PATH, PROCESSED_PATH, workers=workers)

    # Optional: Print how many emails per person
    print("\n📬 Emails per person:")
    for person, df in sorted(person_dfs.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"{person:<20} {len(df)} emails")
