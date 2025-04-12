import pandas as pd
from tqdm import tqdm
from typing import Dict, Any
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

def clean_date_formatting_for_matching(emails_df: pd.DataFrame) -> pd.DataFrame:
    emails_df["date_sent"] = pd.to_datetime(emails_df["date_sent"], errors="coerce")
    emails_df["date_sent"] = emails_df["date_sent"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    emails_df["date_sent"] = emails_df["date_sent"].where(emails_df["date_sent"].notna(), None)
    return emails_df

def create_emails_index(es_client: Elasticsearch, emails_df: pd.DataFrame, index_name: str):
    if es_client.indices.exists(index="emails"):
        es_client.indices.delete(index="emails")

    es_client.indices.create(
        index=index_name,
        body={
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

    actions = [
        {
            "_index": index_name,
            "_source": {
                "subject": row["subject"],
                "body": row["body"],
                "sender": row["sender"],
                "recipients": row["recipients"],
                "cc": row["cc"],
                "date_sent": row["date_sent"],
                "folder": row["folder"]
            }
        }
        for _, row in tqdm(emails_df.iterrows(), total=len(emails_df))
    ]

    bulk(es_client, actions)

def get_rankings_for_query(es_client: Elasticsearch, es_query: Dict[str, Any], num_results_wanted):
    index_name = "emails"
    results = es_client.search(index = index_name, body = es_query, size = num_results_wanted)
    top_emails_with_rankings = results["hits"]["hits"]
    return top_emails_with_rankings


def perform_search(es_client: Elasticsearch, es_query, num_results_wanted: int):
    top_emails_with_rankings = get_rankings_for_query(es_client, es_query, num_results_wanted)
    return top_emails_with_rankings
