from src.utils import load_processed_emails, load_faiss_index
from typing import Optional, List, Dict
from elasticsearch import Elasticsearch
from src.hybrid_search.semantic_search import semantic_search
from src.hybrid_search.combine_rankings import combine_rankings, get_top_emails_by_id
from src.keyword_search.build_es_query import get_persons_to_aliases_dict
from src.keyword_search.es_search import create_emails_index, clean_date_formatting_for_matching, get_keyword_rankings

def hybrid_search(query: str, index, df, es_client, persons_to_aliases_dict, num_results_wanted: int, folder) -> List[Dict]:
    #num_results_each_search = min(max(3*num_results_wanted,100),500)
    num_results_each_search = len(df)
    #semantic_search_results = semantic_search(query, index, df, num_results_each_search)
    semantic_search_results = [(i, 0) for i in range(1, len(df)+1)]

    keyword_search_results = get_keyword_rankings(es_client, query, folder, num_results_each_search, persons_to_aliases_dict)
    return semantic_search_results, keyword_search_results

def get_top_emails(rankings, df, query_len, num_emails, num_results_wanted):
    semantic_rankings, keyword_rankings = rankings
    combined_rankings = combine_rankings(semantic_rankings, keyword_rankings, query_len, num_emails, num_results_wanted)
    top_emails = get_top_emails_by_id(combined_rankings, df)
    print("Email ids and their scores: ")
    print(combined_rankings)
    return top_emails

def send_top_emails_to_file(top_emails, query, fname, folder):
    outfile = open(fname, "a")
    outfile.write("Results in your {} folder for the following query: {}\n\n".format(folder, query))
    for i in range(0,len(top_emails)):
        email = top_emails[i]
        score = email["score"]
        subject = email.get("ExtractedSubject") or "No Subject"
        body = email.get("ExtractedBodyText") or "[No Body Content]"

        outfile.write("Result {}\n".format(i+1,score))
        outfile.write("______________________\n")
        outfile.write("Email ID: {}\n".format(email["Id"]))
        if folder == "inbox":
            outfile.write("From: {}\n".format(email["ExtractedFrom"]))
        else:
            outfile.write("To: {}\n".format(email["ExtractedTo"]))
        outfile.write("CC'd: {}\n".format(email["ExtractedCc"]))
        outfile.write("Date: {}\n".format(email["ExtractedDateSent"]))
        outfile.write("Subject: {}\n".format(subject[:80]))
        outfile.write("Body Preview: {}\n".format(body[:300]))
        outfile.write("\n")
    outfile.close()

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
    
    inbox_df["Id"] = inbox_df.index + 1
    sent_df["Id"] = sent_df.index + 1

    persons_to_aliases_dict = get_persons_to_aliases_dict()

    es_client = Elasticsearch("http://localhost:9200")

    if es_client.ping():
        print("Successfully connected to Elasticsearch.")
    else:
        print("Failed to connect to Elasticsearch.")

    create_emails_index(es_client, inbox_df, "inbox")
    create_emails_index(es_client, sent_df, "sent")

    fname = "top_emails.txt"
    #clear output file before using
    open(fname, "w").close()
    #outfile = open(fname, "a")

    print("Now, you can try some queries.\nBy default, we search the inbox, but you can specify the sent folder.\nEnter '*quit' when prompted for query if you are done.")
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

        top_emails = None
        if folder == "sent":
            num_emails = len(sent_df)
            search_results = hybrid_search(query, sent_index, sent_df, es_client, persons_to_aliases_dict, num_results_wanted, folder)
            top_emails = get_top_emails(search_results, sent_df, len(query), num_emails, num_results_wanted)
        else:
            folder = "inbox"
            num_emails = len(inbox_df)
            search_results = hybrid_search(query, inbox_index, inbox_df, es_client, persons_to_aliases_dict, num_results_wanted, folder)
            top_emails = get_top_emails(search_results, inbox_df, len(query), num_emails, num_results_wanted)
        
        send_top_emails_to_file(top_emails, query, fname, folder)

if __name__ == "__main__":
    main()
