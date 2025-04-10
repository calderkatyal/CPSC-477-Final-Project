import pandas as pd
from tqdm import tqdm
from typing import Dict, Any
from elasticsearch import Elasticsearch

def create_emails_index(es_client: Elasticsearch, emails_df: pd.DataFrame, index_name: str):

    if (not es_client.indices.exists(index = index_name)):
        es_client.indices.create(
            index = index_name,
            body = {
                "settings": {
                    "index": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    }
                },
                "mappings": {
                    "properties": {
                        "subject": {"type": "text"},
                        "body": {"type": "text"},
                        "sender": {"type": "keyword"},
                        "recipients": {"type": "keyword"},
                        "cc": {"type": "keyword"},
                        "date_sent": {"type": "date"},
                        "folder": {"type": "keyword"}
                    }
                }
            }
        )

    for _, row in tqdm(emails_df.iterrows(), total = len(emails_df)):
        email_info = {
            "subject": row["subject"],
            "body": row["body"],
            "sender": row["sender"],
            "recipients": row["recipients"],
            "cc": row["cc"],
            "date_sent": row["date_sent"],
            "folder": row["folder"]
        }
        es_client.index(index = index_name, body = email_info)

def get_rankings_for_query(es_client: Elasticsearch, es_query: Dict[str, Any], index_name: str, num_results_wanted):
    results = es_client.search(index = index_name, body = es_query, size = num_results_wanted)
    top_emails_with_rankings = results["hits"]["hits"]
    return top_emails_with_rankings


def perform_search(es_client: Elasticsearch, es_query, emails_df: pd.DataFrame, num_results_wanted: int):
    index_name = "emails"
    create_emails_index(es_client, emails_df, index_name)
    top_emails_with_rankings = get_rankings_for_query(es_client, es_query, emails_df, index_name, num_results_wanted)
    return top_emails_with_rankings
