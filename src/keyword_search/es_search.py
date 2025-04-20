import pandas as pd
from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from src.keyword_search.build_es_query import build_es_query

def clean_date_formatting_for_matching(emails_df: pd.DataFrame) -> pd.DataFrame:
    emails_df["ExtractedDateSent"] = pd.to_datetime(emails_df["ExtractedDateSent"], errors="coerce")
    emails_df["ExtractedDateSent"] = emails_df["ExtractedDateSent"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    emails_df["ExtractedDateSent"] = emails_df["ExtractedDateSent"].where(emails_df["ExtractedDateSent"].notna(), None)
    return emails_df

def create_emails_index(es_client: Elasticsearch, emails_df: pd.DataFrame, folder_name: str):
    if es_client.indices.exists(index=folder_name):
        es_client.indices.delete(index=folder_name)

    es_client.indices.create(
        index=folder_name,
        body={
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": {
                "properties": {
                    "ExtractedSubject": {"type": "text"},
                    "ExtractedBodyText": {"type": "text"},
                    "ExtractedFrom": {"type": "keyword"},
                    "ExtractedTo": {"type": "keyword"},
                    "ExtractedCc": {"type": "keyword"},
                    "ExtractedDateSent": {"type": "date"},
                    "folder": {"type": "keyword"}
                }
            }
        }
    )

    actions = [
        {
            "_index": folder_name,
            "_id": row["Id"],
            "_source": {
                "subject": row["ExtractedSubject"],
                "body": row["ExtractedBodyText"],
                "sender": row["ExtractedFrom"],
                "recipients": row["ExtractedTo"],
                "cc": row["ExtractedCc"],
                "date_sent": row["ExtractedDateSent"],
                "folder": row["folder"]
            }
        }
        for _, row in emails_df.iterrows()
    ]

    bulk(es_client, actions)

def get_keyword_rankings(es_client: Elasticsearch, query: str, folder_name, num_emails_wanted, persons_to_aliases_dict: Dict[str,List[str]]) -> List[Dict[str, Any]]:
    es_query = build_es_query(query, persons_to_aliases_dict)
    results = es_client.search(index = folder_name, body = es_query, size = num_emails_wanted)
    top_emails_with_rankings = results["hits"]["hits"]
    return top_emails_with_rankings