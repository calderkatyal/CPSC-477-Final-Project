import spacy
import dateparser

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

import spacy
import dateparser

nlp = spacy.load("en_core_web_sm")

def parse_query(query):
    doc = nlp(query)

    # 1. Extract sender (word after "from")
    sender = None
    for i, token in enumerate(doc):
        if token.text.lower() == "from" and i + 1 < len(doc):
            next_token = doc[i + 1]
            if next_token.pos_ in {"PROPN", "NOUN"}:
                sender = next_token.text
                break

    # 2. Extract date range
    dates = [dateparser.parse(ent.text) for ent in doc.ents if "DATE" in ent.label_]
    date_range = None
    if dates and dates[0]:
        start_date = dates[0]
        try:
            end_date = start_date.replace(month=start_date.month + 1)
        except ValueError:
            end_date = None
        date_range = {"start": start_date, "end": end_date}

    # 3. Extract general query keywords
    query_text = " ".join([token.text for token in doc if token.is_alpha and not token.is_stop])

    return {
        "sender": sender,
        "date_range": date_range,
        "query_text": query_text
    }

# Test
query = "emails from Alice in March 2023 about project delta"
parsed = parse_query(query)
print(parsed)

def build_es_query(parsed_query):
    es_query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": parsed_query["query_text"],
                            "fields": ["subject", "body"]
                        }
                    }
                ],
                "filter": []
            }
        }
    }

    # Add sender filter if we have one
    if parsed_query.get("sender"):
        es_query["query"]["bool"]["filter"].append({
            "match": {
                "sender": parsed_query["sender"]
            }
        })

    # Add date range filter if we have one
    if parsed_query.get("date_range"):
        es_query["query"]["bool"]["filter"].append({
            "range": {
                "date_sent": {
                    "gte": parsed_query["date_range"]["start"].strftime("%Y-%m-%d"),
                    "lt": parsed_query["date_range"]["end"].strftime("%Y-%m-%d")
                }
            }
        })

    return es_query

# Test with parsed query
es_query = build_es_query(parsed)
print(es_query)
