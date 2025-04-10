"""
Some testing scripts. (Will provide more elaborate description later).
"""

import os
#import torch
import pandas as pd
#from tqdm import tqdm
from typing import List, Dict, Any
#import numpy as np
from elasticsearch import Elasticsearch
from build_es_query import build_es_query
from es_search import perform_search
from dataloader import load

tqdm.pandas()

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

def get_bm25_rankings(es_client: Elasticsearch, query: str, emails_df: pd.DataFrame, persons_to_aliases, k: int = 10) -> List[Dict[str, Any]]:
    """Perform BM25-based keyword search.

    Args:
        query: Search query
        k: Number of results to return

    Returns:
        List of search results with scores
    """
    # TODO: Implement BM25 search using Elasticsearch
    es_query = build_es_query(query,persons_to_aliasess)
    rankings = perform_search(es_client, es_query, emails_df)
    return rankings

def get_persons_to_aliases_dict():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    RAW_DIR = os.path.join(BASE_DIR, "raw")
    aliases_path = os.path.join(RAW_DIR, "Aliases.csv")
    persons_path = os.path.join(RAW_DIR, "Persons.csv")

    aliases = load(aliases_path)
    persons = load(persons_path)

    aliases['Alias'] = aliases['Alias'].str.lower()
    persons['Person'] = persons['Person'].str.lower()

    #fix this
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
    rankings = get_bm25_rankings(es_client, test_query, emails_df, persons_to_aliases_dict, num_emails_wanted)
    print(rankings)

if __name__ == "__main__":
    main()