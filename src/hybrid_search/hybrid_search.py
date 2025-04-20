from src.utils import load_processed_emails, load_faiss_index
from typing import Optional, List, Dict
from elasticsearch import Elasticsearch
from src.hybrid_search.semantic_search import semantic_search
from src.keyword_search.build_es_query import get_persons_to_aliases_dict
from src.keyword_search.es_search import create_emails_index, clean_date_formatting_for_matching, get_keyword_rankings

def hybrid_search(query: str, index, df, es_client, persons_to_aliases_dict, num_results_wanted: int, folder) -> List[Dict]:
    num_results_each_search = min(max(3*num_results_wanted,100),500)
    #semantic_search_results = semantic_search(query, index, df, num_results_each_search)

    keyword_search_results = get_keyword_rankings(es_client, query, folder, num_results_each_search, persons_to_aliases_dict)

    print("finished search")

def main():
    print("🔄 Loading emails and FAISS index...")
    df = load_processed_emails()

    #inbox_index = load_faiss_index("inbox")
    #sent_index = load_faiss_index("sent")

    inbox_index = None
    sent_index = None

    df = clean_date_formatting_for_matching(df)

    inbox_df = df[df["folder"] == "inbox"].reset_index(drop=True)
    sent_df = df[df["folder"] == "sent"].reset_index(drop=True)

    persons_to_aliases_dict = get_persons_to_aliases_dict()

    es_client = Elasticsearch("http://localhost:9200")

    if es_client.ping():
        print("Successfully connected to Elasticsearch.")
    else:
        print("Failed to connect to Elasticsearch.")

    create_emails_index(es_client, inbox_df, "inbox")
    create_emails_index(es_client, sent_df, "sent")

    print("Now, you can try some queries.\nBy default, we search the inbox, but you can specify the sent folder.\nEnter '*quit' when prompted for query if you are done.")
    #test_query = "Emails about Syria" 
    query = ""
    while True:
        query = input("Query: ")
        if query == '*quit':
            break
        num_results_wanted = input("# of results: ")
        while not num_results_wanted.isdigit():
            num_results_wanted = input("Please enter a positive integer for # of results: ")
        num_results_wanted = int(num_results_wanted)
        folder = input("Folder (inbox/sent): ").lower()
        if folder == "sent":
            hybrid_search(query, sent_index, sent_df, es_client, persons_to_aliases_dict, num_results_wanted, folder)
        else:
            folder = "inbox"
            hybrid_search(query, inbox_index, inbox_df, es_client, persons_to_aliases_dict, num_results_wanted, folder)

if __name__ == "__main__":
    main()
