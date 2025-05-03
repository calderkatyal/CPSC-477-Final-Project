import os
import pandas as pd
import spacy
import dateparser
from typing import Dict, List, Any

nlp = spacy.load("en_core_web_sm")

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
    persons['Person'] = persons['Name'].str.lower()

    persons_map = persons.merge(aliases, right_on='PersonId', left_on='Id', suffixes=('', '_alias'))
    persons_to_aliases = persons_map.groupby('Name')['Alias'].apply(list).to_dict()
    return persons_to_aliases

def parse_query(query: str, persons_to_aliases: Dict[str, List[str]]) -> Dict[str, Any]:
    doc = nlp(query)

    sender_name = None
    for i, token in enumerate(doc):
        if (token.text.lower() == "from") and (i+1 < len(doc)):
            next_token = doc[i+1]
            if (next_token.pos_ == "PROPN") or (next_token.pos_ == "NOUN"):
                sender_name = next_token.text
                break

    sender_aliases = None
    if not(sender_name is None):
        sender_aliases = persons_to_aliases.get(sender_name)

    dates_in_query = [dateparser.parse(ent.text) for ent in doc.ents if "DATE" in ent.label_]
    date_range = None
    if dates_in_query and dates_in_query[0]:
        start_date = dates_in_query[0]
        try:
            end_date = start_date.replace(month = start_date.month + 1)
        except ValueError:
            end_date = None
        date_range = {"start_date": start_date, "end_date": end_date}

    relevant_text = " ".join([token.text for token in doc if (token.is_alpha and not token.is_stop)])
    
    if not relevant_text.strip():
        relevant_text = query.lower().strip()

    query_info = {
        "possible_senders": sender_aliases,
        "date_range": date_range,
        "relevant_text": relevant_text
    }
    return query_info
"""
def build_es_query_from_parsed(parsed_query: Dict[str, Any]) -> Dict[str, Any]:
    es_query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": parsed_query["relevant_text"],
                            "fields": ["subject", "body"]
                        }
                    }
                ],
                "filter": []
            }
        }
    }

    if parsed_query.get("sender"):
        es_query["query"]["bool"]["filter"].append({
            "terms": {"sender": parsed_query["sender"]}
        })

    if parsed_query.get("date_range"):
        start_date = parsed_query["date_range"]["start_date"].strftime("%Y-%m-%d")
        end_date = parsed_query["date_range"]["end_date"].strftime("%Y-%m-%d")
        es_query["query"]["bool"]["filter"].append({
            "range": {
                "date_sent": {"gte": start_date, "lt": end_date}
            }
        })

    return es_query
"""

def build_es_query_from_parsed(parsed_query: Dict[str, Any]) -> Dict[str, Any]:
    es_query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": parsed_query["relevant_text"],
                            "fields": ["subject", "body"]
                        }
                    }
                ],
                "should": [],
                "filter": [] 
            }
        }
    }

    #Boost if sender match
    if parsed_query.get("possible_senders"):
        for sender in parsed_query["possible_senders"]:
            es_query["query"]["bool"]["should"].append({
                "match_phrase": {
                    "ExtractedFrom": {
                        "query": sender,
                        "boost": 2.0
                    }
                }
            })

    #Boost if date within range
    if parsed_query.get("date_range"):
        start_date = parsed_query["date_range"]["start_date"].strftime("%Y-%m-%d")
        end_date = parsed_query["date_range"]["end_date"].strftime("%Y-%m-%d")
        es_query["query"]["bool"]["should"].append({
            "range": {
                "ExtractedDateSent": {
                    "gte": start_date,
                    "lt": end_date,
                    "boost": 1.5
                }
            }
        })

    return es_query


def build_es_query(query: str, persons_to_aliases: Dict[str, List[str]]) -> Dict[str, Any]:
    parsed = parse_query(query, persons_to_aliases)
    es_query = build_es_query_from_parsed(parsed)
    return es_query
