"""
Email preprocessing with parallel processing support.
"""

import pandas as pd
import re
import os
from typing import Dict, List, Tuple
from dataloader import load, save
from datetime import datetime
import email.utils
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import numpy as np

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Register tqdm for pandas
tqdm.pandas()

# Email signoff patterns
SIGNOFF_PATTERNS = [
        # Professional closings
        r"\n\s*Best regards,.*$",
        r"\n\s*Regards,.*$",
        r"\n\s*Sincerely,.*$",
        r"\n\s*Kind regards,.*$",
        r"\n\s*Yours sincerely,.*$",
        r"\n\s*Yours truly,.*$",
        r"\n\s*Yours faithfully,.*$",
        r"\n\s*Respectfully,.*$",
        r"\n\s*Respectfully yours,.*$",
        r"\n\s*Cordially,.*$",
        r"\n\s*Cordially yours,.*$",
        
        # Semi-formal closings
        r"\n\s*Cheers,.*$",
        r"\n\s*Thanks,.*$",
        r"\n\s*Thank you,.*$",
        r"\n\s*Many thanks,.*$",
        r"\n\s*Thanks in advance,.*$",
        r"\n\s*Best,.*$",
        r"\n\s*All the best,.*$",
        r"\n\s*Take care,.*$",
        r"\n\s*Looking forward,.*$",
        r"\n\s*Looking forward to hearing from you,.*$",
        
        # Informal closings
        r"\n\s*Best wishes,.*$",
        r"\n\s*Warm regards,.*$",
        r"\n\s*Warmest regards,.*$",
        r"\n\s*Warm wishes,.*$",
        r"\n\s*Have a great day,.*$",
        r"\n\s*Have a good one,.*$",
        r"\n\s*Talk soon,.*$",
        r"\n\s*See you soon,.*$",
        r"\n\s*Take it easy,.*$",
        
        # Business signoffs (more specific pattern)
        r"\n\s*[A-Z][a-z]+ [A-Z][a-z]+ \| .*$",  # Name | Title/Company
        r"\n\s*[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+ \| .*$",  # First M. Last | Title/Company
        
        # Common name patterns after signoffs
        r"\n\s*[A-Z][a-z]+ [A-Z][a-z]+$",  # First Last
        r"\n\s*[A-Z][a-z]+$",  # Single name
        r"\n\s*[A-Z]\. [A-Z][a-z]+$",  # Initial Last
        r"\n\s*[A-Z][a-z]+ [A-Z]\.$",  # First Initial
        r"\n\s*[A-Z][a-z]+ [A-Z][a-z]+ [A-Z]\.$",  # First Middle Initial
    ]

SIGNOFF_REGEX = re.compile("|".join(SIGNOFF_PATTERNS), flags=re.IGNORECASE|re.MULTILINE)
PUNCTUATION_REGEX = re.compile(r'[^\w\s]')
SUBJECT_CLEAN_REGEX = re.compile(r'^(re|fwd):\s*', flags=re.IGNORECASE)

def parse_headers(txt: str) -> dict:
    """Thread-safe header parsing function."""
    headers = {
        "sender": "Unknown",
        "recipient": "Unknown",
        "cc": "Unknown",
        "subject": "No Subject",
        "date": None,
        "body": txt.strip()
    }

    patterns = {
        "sender": r"^From:\s*(.*)$",
        "recipient": r"^To:\s*(.*)$",
        "cc": r"^Cc:\s*(.*)$",
        "subject": r"^Subject:\s*(.*)$",
        "date": r"^Date:\s*(.*)$"
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, txt, re.MULTILINE)
        if match:
            headers[field] = match.group(1)

    if headers["date"]:
        try:
            date_tuple = email.utils.parsedate_tz(headers["date"])
            if date_tuple:
                headers["date"] = datetime.fromtimestamp(email.utils.mktime_tz(date_tuple))
        except Exception:
            logger.debug(f"Date parse failed: {headers['date']}")

    parts = txt.split("\n\n", 1)
    if len(parts) > 1:
        headers["body"] = parts[1].strip()

    return headers

def clean_text_batch(texts: pd.Series, is_body: bool) -> pd.Series:
    """Vectorized text cleaning for entire columns"""
    if is_body:
        texts = texts.str.replace(SIGNOFF_REGEX, "", regex=True)
        texts = texts.str.lower()
        texts = texts.str.replace(PUNCTUATION_REGEX, " ", regex=True)
        return texts.str.split().str.join(" ")
    return texts.str.replace(SUBJECT_CLEAN_REGEX, "", regex=True).str.lower().str.strip()

def process_chunk(args: Tuple[pd.DataFrame, bool]) -> pd.DataFrame:
    """Process a chunk of emails with full preprocessing."""
    chunk, test_mode = args
    try:
        # Parse headers 
        parsed = chunk["message"].apply(parse_headers).apply(pd.Series)
        chunk = pd.concat([chunk, parsed], axis=1)
        
        chunk["subject"] = clean_text_batch(chunk["subject"], False)
        chunk["body"] = clean_text_batch(chunk["body"], True)
        
        return chunk
    except Exception as e:
        logger.error(f"Error processing chunk: {e}")
        if test_mode:
            raise
        return pd.DataFrame()

def parallel_preprocess(df: pd.DataFrame, workers: int = None, test_mode: bool = False) -> pd.DataFrame:
    """Process DataFrame in parallel chunks."""
    workers = workers or cpu_count() * 2  # Use all available cores
    chunk_size = min(10000, len(df) // workers)  # Optimal chunk size
    
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        chunks = np.array_split(df, workers * 4)  # More chunks for better load balancing
        futures = {
            executor.submit(process_chunk, (chunk, test_mode)): i
            for i, chunk in enumerate(chunks)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            results.append(future.result())
    
    return pd.concat(results, ignore_index=True)

def organize_by_person(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Organize processed emails by person."""
    person_dfs = {}
    all_people = pd.concat([df["sender"], df["recipient"]]).dropna().unique()
    
    for person in tqdm(all_people, desc="Organizing by person"):
        if person == "Unknown":
            continue
            
        emails = df[
            (df["sender"] == person) | 
            (df["recipient"] == person) |
            (df["cc"].str.contains(person, na=False))
        ].copy()
        
        emails["direction"] = emails["sender"].apply(
            lambda x: "sent" if x == person else "received"
        )
        
        if not emails.empty:
            person_dfs[person] = emails
    
    return person_dfs

def preprocess_emails(input_path: str, output_path: str, workers: int = None) -> Dict[str, pd.DataFrame]:
    """Main preprocessing pipeline with parallel processing."""
    logger.info("Loading raw emails...")
    df = load(input_path)
    
    logger.info("Starting parallel processing...")
    processed_df = parallel_preprocess(df, workers=workers)
    
    logger.info("Organizing by person...")
    person_dfs = organize_by_person(processed_df)
    
    logger.info("Saving processed data...")
    save(processed_df, output_path)
    
    logger.info(f"Processed {len(person_dfs)} people with {len(processed_df)} total emails")
    return person_dfs

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    RAW_CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "emails.csv")
    PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "emails.parquet")
    
    workers = cpu_count() * 2  # Utilize all CPU cores
    preprocess_emails(RAW_CSV_PATH, PROCESSED_PATH, workers=workers)