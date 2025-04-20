from src.utils import load_processed_emails, load_faiss_index
from typing import List, Dict
from elasticsearch import Elasticsearch
from src.hybrid_search.semantic_search import semantic_search
from src.hybrid_search.combine_rankings import combine_rankings, get_top_emails_by_id
from src.keyword_search.build_es_query import get_persons_to_aliases_dict
from src.keyword_search.es_search import create_emails_index, clean_date_formatting_for_matching, get_keyword_rankings

def hybrid_search(query: str, index, df, es_client, persons_to_aliases_dict, folder: str, search_mode: str):
    num_results_each_search = len(df)

    semantic_search_results = []
    keyword_search_results = []

    if search_mode in {"hybrid", "semantic"}:
        semantic_search_results = semantic_search(query, index, df)

    if search_mode in {"hybrid", "keyword"}:
        keyword_search_results = get_keyword_rankings(
            es_client, query, folder, num_results_each_search, persons_to_aliases_dict
        )

    return semantic_search_results, keyword_search_results

def get_top_emails(rankings, df, query_len, num_emails, num_results_wanted):
    semantic_rankings, keyword_rankings = rankings
    combined_rankings = combine_rankings(semantic_rankings, keyword_rankings, query_len, num_emails, num_results_wanted)
    top_emails = get_top_emails_by_id(combined_rankings, df)
    return top_emails

def send_top_emails_to_file(top_emails, query, fname, folder):
    with open(fname, "a") as outfile:
        outfile.write("Results in your {} folder for the following query: {}\n\n".format(folder, query))
        for i, email in enumerate(top_emails):
            score = email["score"]
            subject = email.get("ExtractedSubject") or "No Subject"
            body = email.get("ExtractedBodyText") or "[No Body Content]"
            outfile.write("Result {}\n".format(i+1, score))
            outfile.write("______________________\\n")
            outfile.write("Email ID: {}\n".format(email["Id"]))
            if folder == "inbox":
                outfile.write("From: {}\n".format(email["ExtractedFrom"]))
            else:
                outfile.write("To: {}\n".format(email["ExtractedTo"]))
            outfile.write("CC'd: {}\n".format(email["ExtractedCc"]))
            outfile.write("Date: {}\n".format(email["ExtractedDateSent"]))
            outfile.write("Subject: {}\n".format(subject[:80]))
            outfile.write("Body Preview: {}\n\n".format(body[:300]))
    print("Added output to {}".format(fname))

def main():
    print("🔄 Loading emails and FAISS index...")
    df = load_processed_emails()

    inbox_index = load_faiss_index("inbox")
    sent_index = load_faiss_index("sent")

    df = clean_date_formatting_for_matching(df)
    inbox_df = df[df["folder"] == "inbox"].reset_index(drop=True)
    sent_df = df[df["folder"] == "sent"].reset_index(drop=True)
    inbox_df["Id"] = inbox_df.index + 1
    sent_df["Id"] = sent_df.index + 1

    persons_to_aliases_dict = get_persons_to_aliases_dict()
    es_client = Elasticsearch("http://localhost:9200")

    if not es_client.ping():
        print("❌ Failed to connect to Elasticsearch.")
        return

    create_emails_index(es_client, inbox_df, "inbox")
    create_emails_index(es_client, sent_df, "sent")

    fname = "top_emails.txt"
    open(fname, "w").close()

    print("Now, you can try some queries.")
    print("Enter '*quit' when prompted for query if you are done.")
    while True:
        query = input("Query: ")
        if query.strip() == "*quit":
            break

        num_results_wanted = input("# of results: ")
        while not num_results_wanted.isdigit():
            num_results_wanted = input("Please enter a positive integer for # of results: ")
        num_results_wanted = int(num_results_wanted)

        folder = input("Folder (inbox/sent): ").lower()
        if folder not in {"inbox", "sent"}:
            folder = "inbox"

        search_mode = input("Search mode (hybrid / semantic / keyword): ").lower()
        while search_mode not in {"hybrid", "semantic", "keyword"}:
            search_mode = input("Please enter valid mode (hybrid / semantic / keyword): ").lower()

        df_used = inbox_df if folder == "inbox" else sent_df
        index = inbox_index if folder == "inbox" else sent_index
        num_emails = len(df_used)

        rankings = hybrid_search(query, index, df_used, es_client, persons_to_aliases_dict, folder, search_mode)
        top_emails = get_top_emails(rankings, df_used, len(query), num_emails, num_results_wanted)

        send_top_emails_to_file(top_emails, query, fname, folder)

if __name__ == "__main__":
    main()