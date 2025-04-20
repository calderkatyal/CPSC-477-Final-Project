from src.utils import load_processed_emails, load_faiss_index
from typing import Optional, List, Dict
from elasticsearch import Elasticsearch
from src.hybrid_search.semantic_search import semantic_search
from src.keyword_search.build_es_query import get_persons_to_aliases_dict
from src.keyword_search.es_search import create_emails_index, clean_date_formatting_for_matching, get_keyword_rankings
import statistics #I believe numpy has a compatibility issue with spacy, so using this since it doesn't matter that much I think
import heapq

def get_z_scores(scores):
    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores)
    if stdev == 0:
        return [0 for x in scores]
    return [(x - mean) / stdev for x in scores]

def get_average_top_z_score(scores):
    num_top_to_take = 5
    avg_top_z_score = statistics.mean(scores[:num_top_to_take])
    return avg_top_z_score

def insert_zeros_missing_ids(rankings, num_emails):
    rankings_filled = [None] * num_emails
    num_results = len(rankings)
    j = 0
    for i in range(1,num_emails+1):
        while j < num_results:
            ID, score = rankings[j]
            if ID == i:
                rankings_filled[i-1] = (ID, score)
                j+=1
                break
            elif ID > i:
                rankings_filled[i-1] = (i, 0)
                break
            j+=1
        if j == num_results:
            rankings_filled[i-1] = (i, 0)
    return rankings_filled


def combine_rankings(semantic_rankings, keyword_rankings, query_len, num_emails, num_results_wanted):
    semantic_scores = [score for _, score in semantic_rankings]
    keyword_scores = [score for  _, score in keyword_rankings]
    
    #pad keyword scores (elastic search does not return a result if it does not match any words at all in the query)
    #also if we have a ton of emails, we only ask the keyword search for 500
    keyword_scores += [0] * (len(semantic_scores) - len(keyword_scores))

    semantic_z_scores = get_z_scores(semantic_scores)
    keyword_z_scores = get_z_scores(keyword_scores)

    #how much better the top scores really are compared to the rest (did we discriminate well)
    average_top_z_score_semantic = get_average_top_z_score(semantic_scores)
    average_top_z_score_keyword = get_average_top_z_score(keyword_scores)

    semantic_bias = 2 #assuming semantic search is better, value it twice as much
    if query_len <= 4:
        semantic_bias = 1 #if query very short, semantic search probably won't be much better - in fact, keyword search might actually be better
    elif query_len <= 8:
        semantic_bias = semantic_bias * (2/3) #if pretty short, lower bias toward semantic search somewhat
    if average_top_z_score_semantic < (average_top_z_score_keyword / 5):
        semantic_bias = semantic_bias * (1/2)
    elif average_top_z_score_semantic > average_top_z_score_keyword:
        semantic_bias * (3/2)

    #fill in scores of zero for missing email IDs, in order to allow for much faster ranking combination 
    semantic_scores_filled = insert_zeros_missing_ids(semantic_rankings, num_emails)
    keyword_scores_filled = insert_zeros_missing_ids(keyword_rankings, num_emails)
    combined_scores = [None] * num_emails

    for i in range(0, num_emails):
        ID, sem_score = semantic_scores_filled[i]
        _, kw_score = keyword_scores_filled[i]
        combined_score = (sem_score * semantic_bias) + kw_score
        combined_scores[i] = (ID, combined_score)

    top_results = heapq.nlargest(num_results_wanted, combined_scores, key=lambda x: x[1])
    return top_results

def get_top_emails_by_id(top_results, df):
    top_emails = []
    for ID, sem_score in top_results:
     #   print(str(ID))
      #  print(df[df["Id"] == ID])
        email = df[df["Id"] == ID].iloc[0]
        email["score"] = sem_score
        top_emails.append(email)
    return top_emails

def main():
    num_emails = 10
    semantic_rankings = [None] * 5
    keyword_rankings = [None] * 5
    for i in range(0, 5):
        semantic_rankings[i] = ((i+1)*2,0)
        keyword_rankings[i] = ((i+1)*2+1,0)
    query_len = 10
    num_results_wanted = 8
    combined_rankings = combine_rankings(semantic_rankings, keyword_rankings, query_len, num_emails, num_results_wanted)
    print(combined_rankings)

if __name__ == "__main__":
    main()