"""
Email preprocessing utilities for preparing data for embedding generation.
"""
import pandas as pd
import re
import spacy
import os
from typing import Dict, List
from dataloader import load, save
from datetime import datetime
import email.utils
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simplified format
)
logger = logging.getLogger(__name__)

# Initialize spaCy for text processing
logger.info("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

def parse_headers(txt: str) -> dict:
    """Parse email headers and body from raw text."""
    flags = re.MULTILINE
    s = re.search(r"^From:\s*(.*)$", txt, flags)
    r = re.search(r"^To:\s*(.*)$", txt, flags)
    c = re.search(r"^Cc:\s*(.*)$", txt, flags)
    sbj = re.search(r"^Subject:\s*(.*)$", txt, flags)
    date_str = re.search(r"^Date:\s*(.*)$", txt, flags)
    
    # Parse date if present
    date = None
    if date_str:
        try:
            date_tuple = email.utils.parsedate_tz(date_str.group(1))
            if date_tuple:
                date = datetime.fromtimestamp(email.utils.mktime_tz(date_tuple))
        except Exception as e:
            logger.warning(f"Could not parse date: {date_str.group(1)}")
    
    parts = txt.split("\n\n", 1)
    b = parts[1].strip() if len(parts) > 1 else txt.strip()
    return {
        "sender": s.group(1) if s else "Unknown",
        "recipient": r.group(1) if r else "Unknown",
        "cc": c.group(1) if c else "Unknown",
        "subject": sbj.group(1) if sbj else "No Subject",
        "date": date,
        "body": b
    }

def clean_body(body: str) -> str:
    """Cleans body of email for embedding generation."""
    # Common signoffs that only appear at the end
    signoffs = [
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
    
    # Remove signoffs from the end of the email
    for signoff in signoffs:
        body = re.sub(signoff, "", body, flags=re.IGNORECASE | re.MULTILINE)
    
    # Process with spaCy
    doc = nlp(body)
    
    # Keep only meaningful tokens (remove stopwords, punctuation, and whitespace)
    cleaned_tokens = [
        token.lemma_.lower() 
        for token in doc 
        if not (token.is_stop or token.is_punct or token.is_space)
    ]
    
    return " ".join(cleaned_tokens)

def clean_subject(subject: str) -> str:
    """Cleans subject line of email."""
    subject = re.sub(r"^(re|fwd):", "", subject.lower())
    return subject.strip()

def preprocess_emails(input_path: str, output_path: str) -> Dict[str, pd.DataFrame]:
    """Preprocess emails and organize by person."""
    # Load and parse emails
    df = load(input_path)
    logger.info(f"Processing {len(df)} emails...")
    
    # Parse headers with progress bar
    logger.info("Parsing headers...")
    df_parsed = df["message"].progress_apply(parse_headers).apply(pd.Series)
    df = df.join(df_parsed)
    
    # Clean text fields with progress bars
    logger.info("Cleaning subjects...")
    df["subject"] = df["subject"].progress_apply(clean_subject)
    
    logger.info("Cleaning bodies...")
    df["body"] = df["body"].progress_apply(clean_body)
    
    # Sort by date if available
    if "date" in df.columns:
        df = df.sort_values("date")
    
    # Create dictionary of dataframes by person
    person_dfs = {}
    
    # Process both senders and recipients
    all_people = pd.concat([df["sender"], df["recipient"]]).unique()
    logger.info(f"Found {len(all_people)} unique people")
    
    # Process each person with progress bar
    logger.info("Organizing emails by person...")
    for person in tqdm(all_people, desc="Processing people"):
        if person == "Unknown":
            continue
            
        # Get all emails where this person is either sender or recipient
        person_emails = df[
            (df["sender"] == person) | 
            (df["recipient"] == person) |
            (df["cc"].str.contains(person, na=False))
        ].copy()
        
        # Add direction column (sent/received)
        person_emails["direction"] = person_emails.apply(
            lambda row: "sent" if row["sender"] == person else "received",
            axis=1
        )
        
        if not person_emails.empty:
            person_dfs[person] = person_emails
    
    # Save the processed data
    save(df, output_path)
    logger.info(f"Processed {len(person_dfs)} people's emails")
    
    return person_dfs

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    RAW_CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "emails.csv")
    PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "emails.parquet")
    
    person_dfs = preprocess_emails(RAW_CSV_PATH, PROCESSED_PATH)