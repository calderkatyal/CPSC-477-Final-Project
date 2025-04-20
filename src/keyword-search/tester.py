"""
Some testing scripts. (Will provide more elaborate description later).
"""

import os
import pandas as pd
from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from build_es_query import get_persons_to_aliases_dict, build_es_query
from es_search import create_emails_index, clean_date_formatting_for_matching, perform_search
from sqlalchemy import create_engine

def load_emails_from_sql() -> pd.DataFrame:
    DB_URL = "postgresql://postgres:password@localhost:5432/emails_db"
    engine = create_engine(DB_URL)
    query = "SELECT * FROM emails"
    emails_df = pd.read_sql(query, engine)
    return emails_df

def get_bm25_rankings(es_client: Elasticsearch, query: str, persons_to_aliases_dict: Dict[str,List[str]], num_emails_wanted: int = 10) -> List[Dict[str, Any]]:
    es_query = build_es_query(query, persons_to_aliases_dict)
    top_emails_with_rankings = perform_search(es_client, es_query, num_emails_wanted)
    return top_emails_with_rankings

def main():
    emails_df = load_emails_from_sql()
    emails_df = clean_date_formatting_for_matching(emails_df)

    persons_to_aliases_dict = get_persons_to_aliases_dict()

    es_client = Elasticsearch("http://localhost:9200")

    if es_client.ping():
        print("Successfully connected to Elasticsearch.")
    else:
        print("Failed to connect to Elasticsearch.")

    print("Creating Elasticquery index for keyword search. After this, we'll be ready for any queries.")
    index_name = "emails"
    create_emails_index(es_client, emails_df, index_name)
    print("Now, you can try some queries. Enter '*quit' if you are done.")
    #test_query = "Emails about Syria" 
    query = ""
    while True:
        query = input("Query: ")
        if query == '*quit':
            break
        print("    Searching...")
        num_emails_wanted = 10
        top_emails_with_rankings = get_bm25_rankings(es_client, query, persons_to_aliases_dict, num_emails_wanted)
        print("    Results: ")
        for result in top_emails_with_rankings:
            #print(result['_source'])
            print(result)
        #print(top_emails_with_rankings)

if __name__ == "__main__":
    main()