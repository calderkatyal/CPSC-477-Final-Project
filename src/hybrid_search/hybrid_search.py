from src.utils import load_processed_emails, load_faiss_index
from typing import List, Dict
from elasticsearch import Elasticsearch
from src.semantic_search.semantic_search import semantic_search
from src.hybrid_search.hybrid_rankings import combine_rankings, get_top_emails_by_id
from src.keyword_search.build_es_query import get_persons_to_aliases_dict
from src.keyword_search.es_search import create_emails_index, clean_date_formatting_for_matching, get_keyword_rankings
from src.query_expansion.rrf_fusion import reciprocal_rank_fusion
from src.evaluation.metrics import weighted_consistency_top_k, weighted_kendalls_w, weighted_pairwise_mse
from src.semantic_search.semantic_search import init_semantic_components 
import heapq

def safe_input(prompt: str) -> str:
    val = input(prompt)
    if val.strip() == "*quit":
        print("👋 Exiting search.")
        exit(0)
    return val

def hybrid_search(query: str, index, df, es_client, persons_to_aliases_dict, folder: str, search_mode: str):
    num_results_each_search = len(df)
    semantic_search_results = []
    keyword_search_results = []

    if search_mode in {"hybrid", "semantic"}:
        semantic_variants = semantic_search(query, index, df)
        semantic_search_results = reciprocal_rank_fusion(semantic_variants) 
        semantic_search_results = sorted(semantic_search_results, key=lambda x: x[0])

    if search_mode in {"hybrid", "keyword"}:
        keyword_search_results = get_keyword_rankings(
            es_client, query, folder, num_results_each_search, persons_to_aliases_dict
        )

    return semantic_search_results, keyword_search_results

def get_top_emails(rankings, df, query, query_len, num_emails, num_results_wanted, is_test=False):
    semantic_rankings, keyword_rankings = rankings
    combined_rankings = combine_rankings(semantic_rankings, keyword_rankings, query_len, num_emails, num_results_wanted, is_test=is_test)
    top_emails = get_top_emails_by_id(combined_rankings, df)
    return top_emails

def get_best_emails_across_queries(ranked_emails):
    inverted = [
        [(-float(email["score"]), str(email["Id"]), email) for email in lst]
        for lst in ranked_emails
    ]

    merged = heapq.merge(*inverted)

    seen_emails = set()
    best_emails = []

    k = 4
    for _, id_str, item in merged:
        if id_str not in seen_emails:
            seen_emails.add(id_str)
            best_emails.append(item)
            if len(best_emails) == k:
                break

    return best_emails

def send_top_emails_across_queries_to_file(top_emails, queries, fname, folder, query_set_count):
    with open(fname, "a") as outfile:
        outfile.write(f"\n~~~~~~~~~~~~ QUERY SET #{query_set_count} ~~~~~~~~~~~~\n")
        outfile.write("Folder: {}\n".format(folder))
        for i, query in enumerate(queries):
            outfile.write("Query {}: {}\n".format(i+1, query))
        outfile.write("Top 4 distinct results across queries: \n")
        for i, email in enumerate(top_emails):
            score = email["score"]
            subject = email.get("ExtractedSubject") or "No Subject"
            body = email.get("ExtractedBodyText") or "[No Body Content]"
            outfile.write("Result {}\n".format(i+1))
            outfile.write("______________________\n")
            outfile.write("Email ID: {}\n".format(email["Id"]))
            if folder == "inbox":
                outfile.write("From: {}\n".format(email["ExtractedFrom"]))
            else:
                outfile.write("To: {}\n".format(email["ExtractedTo"]))
            outfile.write("CC'd: {}\n".format(email["ExtractedCc"]))
            outfile.write("Date: {}\n".format(email["ExtractedDateSent"]))
            outfile.write("Subject: {}\n".format(subject[:80]))
            outfile.write("Body Preview: {}\n\n".format(body[:1000]))
    print(f"✅ Added output to {fname}")

def send_top_emails_to_file(top_emails, query, fname, folder, query_count):
    with open(fname, "a") as outfile:
        outfile.write(f"\n~~~~~~~~~~~~ QUERY #{query_count} ~~~~~~~~~~~~\n")
        outfile.write("Folder: {}\nQuery: {}\n\n".format(folder, query))
        for i, email in enumerate(top_emails):
            score = email["score"]
            subject = email.get("ExtractedSubject") or "No Subject"
            body = email.get("ExtractedBodyText") or "[No Body Content]"
            outfile.write("Result {}\n".format(i+1, score))
            outfile.write("______________________\n")
            outfile.write("Email ID: {}\n".format(email["Id"]))
            if folder == "inbox":
                outfile.write("From: {}\n".format(email["ExtractedFrom"]))
            else:
                outfile.write("To: {}\n".format(email["ExtractedTo"]))
            outfile.write("CC'd: {}\n".format(email["ExtractedCc"]))
            outfile.write("Date: {}\n".format(email["ExtractedDateSent"]))
            outfile.write("Subject: {}\n".format(subject[:80]))
            outfile.write("Body Preview: {}\n\n".format(body[:1000]))
    print(f"✅ Added output to {fname}")

def run_search_interface(is_test=False, seed: int=None):
    print("🛠️ Initializing semantic components...")
    init_semantic_components(seed=seed)
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
    fname_test = "top_across_queries.txt"
    if is_test:
        open(fname_test, "w").close()
    else:
        open(fname, "w").close()
    query_count = 1

    print("Now, you can try some queries.")
    print("Enter '*quit' at any prompt to exit.")

    while True:
        if is_test:
            query1 = safe_input("Query 1: ")
            query2 = safe_input("Query 2: ")
            query3 = safe_input("Query 3: ")
            query4 = safe_input("Query 4: ")
            folder = safe_input("Folder (inbox/sent): ").lower()
            search_mode = safe_input("Search mode (hybrid / semantic / keyword): ").lower()
            while search_mode not in {"hybrid", "semantic", "keyword"}:
                search_mode = safe_input("Please enter valid mode (hybrid / semantic / keyword): ").lower()
            df_used = inbox_df if folder == "inbox" else sent_df
            index = inbox_index if folder == "inbox" else sent_index
            num_emails = len(df_used)

            rankings1 = hybrid_search(query1, index, df_used, es_client, persons_to_aliases_dict, folder, search_mode)
            rankings2 = hybrid_search(query2, index, df_used, es_client, persons_to_aliases_dict, folder, search_mode)
            rankings3 = hybrid_search(query3, index, df_used, es_client, persons_to_aliases_dict, folder, search_mode)
            rankings4 = hybrid_search(query4, index, df_used, es_client, persons_to_aliases_dict, folder, search_mode)
            
            top_emails1 = get_top_emails(rankings1, df_used, query1, len(query1.strip().split()), num_emails, -1, is_test)
            top_emails1_info  = [{"Id": int(email["Id"]), "score": email["score"]} for email in top_emails1]

            top_emails2 = get_top_emails(rankings2, df_used, query2, len(query2.strip().split()), num_emails, -1, is_test)
            top_emails2_info  = [{"Id": int(email["Id"]), "score": email["score"]} for email in top_emails2]

            top_emails3 = get_top_emails(rankings3, df_used, query3, len(query3.strip().split()), num_emails, -1, is_test)
            top_emails3_info  = [{"Id": int(email["Id"]), "score": email["score"]} for email in top_emails3]

            top_emails4 = get_top_emails(rankings4, df_used, query4, len(query4.strip().split()), num_emails, -1, is_test)
            top_emails4_info  = [{"Id": int(email["Id"]), "score": email["score"]} for email in top_emails4]

            emails = [top_emails1_info, top_emails2_info, top_emails3_info, top_emails4_info]
            
            queries = [query1, query2, query3, query4]
            best_emails_across = get_best_emails_across_queries([top_emails1, top_emails2, top_emails3, top_emails4])
            send_top_emails_across_queries_to_file(best_emails_across, queries, fname_test, folder, query_count)
            wkw = weighted_kendalls_w(emails)
            wmse = weighted_pairwise_mse(emails)
            ctk = weighted_consistency_top_k(emails)
            print(f"Weighted MSE (0 = high agreement): {wmse:.3f}")
            print(f"Weighted Kendall's W (1 = high agreement): {wkw:.3f}")
        else: 
            query = safe_input("Query: ")

            num_results_wanted = safe_input("# of results: ")
            while not num_results_wanted.isdigit():
                num_results_wanted = safe_input("Please enter a positive integer for # of results: ")
            num_results_wanted = int(num_results_wanted)

            folder = safe_input("Folder (inbox/sent): ").lower()
            while folder not in {"inbox", "sent"}:
                folder = safe_input("Please enter valid folder (inbox/sent): ").lower()

            search_mode = safe_input("Search mode (hybrid / semantic / keyword): ").lower()
            while search_mode not in {"hybrid", "semantic", "keyword"}:
                search_mode = safe_input("Please enter valid mode (hybrid / semantic / keyword): ").lower()

            df_used = inbox_df if folder == "inbox" else sent_df
            index = inbox_index if folder == "inbox" else sent_index
            num_emails = len(df_used)

            rankings = hybrid_search(query, index, df_used, es_client, persons_to_aliases_dict, folder, search_mode)
            top_emails = get_top_emails(rankings, df_used, query, len(query.strip().split()), num_emails, num_results_wanted, is_test)
            send_top_emails_to_file(top_emails, query, fname, folder, query_count)
        query_count += 1
