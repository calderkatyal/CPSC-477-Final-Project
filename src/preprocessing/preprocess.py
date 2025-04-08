import pandas as pd
import re
import os
from typing import Tuple
from dataloader import load, save
from datetime import datetime
from tqdm import tqdm
import logging
from multiprocessing import cpu_count

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
tqdm.pandas()

# Compile regex patterns
PUNCTUATION_REGEX = re.compile(r'[^\w\s]')
SUBJECT_CLEAN_REGEX = re.compile(r'^(re|fwd):\s*', flags=re.IGNORECASE)
HEADER_CLEAN_REGEX = re.compile(r'^.*?From')
REPEATED_HEADER_CLEAN_REGEX = re.compile(r'UNCLASSIFIED.*?STATE.*\n')

def extract_alias(raw_from: str) -> str:
    """Extract the alias or email from the 'From' field."""
    if pd.isnull(raw_from):
        return ''
    raw_from = raw_from.lower().strip()
    match = re.search(r'<(.*?)>', raw_from)
    if match:
        return match.group(1).strip()
    match = re.search(r'[\w\.-]+@[\w\.-]+', raw_from)
    if match:
        return match.group(0).strip()
    return raw_from

def clean_text(texts: pd.Series, is_body: bool) -> pd.Series:
    texts = texts.fillna("").astype(str)
    if is_body:
        texts = texts.progress_apply(lambda x: re.sub(HEADER_CLEAN_REGEX,"",x))
        texts = texts.progress_apply(lambda x: re.sub(REPEATED_HEADER_CLEAN_REGEX,"",x))
    texts = texts.progress_apply(lambda x: re.sub(PUNCTUATION_REGEX, " ", x.lower()))
    texts = texts.str.split().progress_apply(lambda words: ' '.join(words))
    if not is_body:
        texts = texts.progress_apply(lambda x: re.sub(SUBJECT_CLEAN_REGEX, "", x).strip())
    return texts

def preprocess_emails(
    emails_path: str,
    receivers_path: str,
    aliases_path: str,
    persons_path: str,
    output_path_base: str,
    workers: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    logger.info("Loading Hillary Clinton dataset...")
    emails = load(emails_path)
    receivers = load(receivers_path)
    aliases = load(aliases_path)
    persons = load(persons_path)

    logger.info(f"Loaded {len(emails)} emails from {emails_path}")
    logger.info(f"Loaded {len(receivers)} receivers from {receivers_path}")
    logger.info(f"Loaded {len(aliases)} aliases from {aliases_path}")
    logger.info(f"Loaded {len(persons)} persons from {persons_path}")

    logger.info("Merging aliases with persons...")
    alias_map = aliases.merge(persons, left_on='PersonId', right_on='Id', suffixes=('', '_person'))
    alias_map['Alias'] = alias_map['Alias'].str.lower()
    alias_to_name = dict(zip(alias_map['Alias'], alias_map['Name']))

    logger.info("Extracting cleaned aliases from sender field...")
    emails['CleanedAlias'] = emails['ExtractedFrom'].progress_apply(extract_alias)
    emails['SenderName'] = emails['CleanedAlias'].map(alias_to_name)

    # 🛠 Fill missing ExtractedFrom using CleanedAlias
    emails['ExtractedFrom'] = emails['ExtractedFrom'].fillna(emails['CleanedAlias'])

    logger.info("Cleaning subject and body...")
    emails['subject'] = clean_text(emails['ExtractedSubject'], is_body=False)
    emails['body'] = clean_text(emails['ExtractedBodyText'], is_body=True)

    logger.info("Parsing dates...")
    emails['date'] = emails['ExtractedDateSent'].progress_apply(
    lambda date_str: pd.to_datetime(
        re.sub(r'\bOM\b', 'AM', re.sub(r'\b[A-Z]{1,3}$', '', date_str)) if isinstance(date_str, str) else date_str,
        errors='coerce'
    )


    )

    logger.info("Identifying Hillary aliases and PersonId...")
    hillary_aliases = alias_map[alias_map['Name'].str.contains("hillary", case=False, na=False)]
    hillary_alias_set = set(hillary_aliases['Alias'])
    hillary_person_ids = set(
        persons[persons['Name'].str.contains("hillary", case=False, na=False)]['Id']
    )

    logger.info("Flagging sent emails...")
    emails['IsHillarySender'] = emails['CleanedAlias'].isin(hillary_alias_set)
    sent_df = emails[emails['IsHillarySender']]

    logger.info("Flagging received emails...")
    receivers = receivers.merge(persons, left_on='PersonId', right_on='Id')
    received_email_ids = set(
        receivers[receivers['PersonId'].isin(hillary_person_ids)]['EmailId']
    )
    inbox_df = emails[emails['Id'].isin(received_email_ids)]

    # 🛠 Reconstruct ExtractedTo from receivers
    logger.info("Reconstructing ExtractedTo from receivers...")
    email_to_recipients = (
        receivers.groupby('EmailId')['Name']
        .apply(list)
        .to_dict()
    )
    emails['ExtractedTo'] = emails['Id'].map(email_to_recipients)

    # Re-assign to inbox/sent with updated ExtractedTo
    inbox_df = emails[emails['Id'].isin(received_email_ids)]
    sent_df = emails[emails['IsHillarySender']]

    logger.info(f"Inbox: {len(inbox_df)} emails | Sent: {len(sent_df)} emails")

    logger.info("Saving processed data...")
    inbox_path = os.path.join(output_path_base, "Inbox.parquet")
    sent_path = os.path.join(output_path_base, "Sent.parquet")

    columns_to_keep = [
        "Id",
        "ExtractedSubject",
        "ExtractedBodyText",
        "ExtractedFrom",
        "ExtractedTo",
        "ExtractedCc",           
        "ExtractedDateSent"
    ]

    inbox_df = inbox_df[columns_to_keep]
    sent_df = sent_df[columns_to_keep]

    # Remove rows with missing Extracted Body Text
    inbox_df = inbox_df[inbox_df['ExtractedBodyText'].notnull()]
    sent_df = sent_df[sent_df['ExtractedBodyText'].notnull()]


    save(inbox_df, inbox_path)
    save(sent_df, sent_path)

    return inbox_df, sent_df

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    RAW_DIR = os.path.join(BASE_DIR, "raw")
    PROCESSED_PATH = os.path.join(BASE_DIR, "processed")

    emails_path = os.path.join(RAW_DIR, "Emails.csv")
    receivers_path = os.path.join(RAW_DIR, "EmailReceivers.csv")
    aliases_path = os.path.join(RAW_DIR, "Aliases.csv")
    persons_path = os.path.join(RAW_DIR, "Persons.csv")

    workers = cpu_count() * 2
    inbox_df, sent_df = preprocess_emails(
        emails_path,
        receivers_path,
        aliases_path,
        persons_path,
        PROCESSED_PATH,
        workers=workers
    )

    print(f"\n📥 Nonempty Inbox emails: {len(inbox_df)}")
    print(f"📤 Nonempty Sent emails:  {len(sent_df)}")

