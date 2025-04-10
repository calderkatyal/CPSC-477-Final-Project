"""
Some testing scripts. (Will provide more elaborate description later).
"""

import os
import pandas as pd
from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from build_es_query import build_es_query
from es_search import perform_search

def load_emails() -> pd.DataFrame:
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

def get_bm25_rankings(es_client: Elasticsearch, query: str, emails_df: pd.DataFrame, persons_to_aliases, num_emails_wanted: int = 10) -> List[Dict[str, Any]]:
    es_query = build_es_query(query,persons_to_aliases)
    top_emails_with_rankings = perform_search(es_client, es_query, emails_df, k)
    return top_emails_with_rankings

def get_persons_to_aliases_dict():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    RAW_DIR = os.path.join(BASE_DIR, "raw")
    aliases_path = os.path.join(RAW_DIR, "Aliases.csv")
    persons_path = os.path.join(RAW_DIR, "Persons.csv")

    if not os.path.exists(aliases_path):
        raise FileNotFoundError(f"File not found: {aliases_path}")
    if not os.path.exists(persons_path):
        raise FileNotFoundError(f"File not found: {persons_path}")
    aliases = pd.read_csv(aliases_path)
    persons = pd.read_csv(persons_path)

    aliases['Alias'] = aliases['Alias'].str.lower()
    persons['Person'] = persons['Person'].str.lower()

    persons_map = persons.merge(aliases, right_on='PersonId', left_on='Id', suffixes=('', '_alias'))
    persons_to_aliases = persons_map.groupby('Person')['Alias_alias'].apply(list).to_dict()
    return persons_to_aliases

def main():
    print("📥 Loading processed emails...")
    emails_df = load_emails()
    print(f"Loaded {len(emails_df)} emails.")

    persons_to_aliases_dict = get_persons_to_aliases_dict()

    es_client = Elasticsearch("http://localhost:9200")

    if es_client.ping():
        print("Successfully connected to Elasticsearch.")
    else:
        print("Failed to connect to Elasticsearch.")

    test_query = "" #modify to allow command-line input
    num_emails_wanted = 10
    top_emails_with_rankings = get_bm25_rankings(es_client, test_query, emails_df, persons_to_aliases_dict, num_emails_wanted)
    print(top_emails_with_rankings)

if __name__ == "__main__":
    main()