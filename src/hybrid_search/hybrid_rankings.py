import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Tuple
import heapq
import math
import statistics

model_name = "gpt2" 
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def compute_perplexity(query):
    query_with_context = f"Email content: {query}"
    inputs = tokenizer(query_with_context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels = inputs["input_ids"])
        loss = outputs.loss
    return math.exp(loss.item())

def min_max_normalize(scores: List[float]) -> List[float]:
    """
    Normalize a list of scores using min-max scaling to [0, 1].

    Args:
        scores: Raw float scores.

    Returns:
        List of normalized scores.
    """
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [0.0 for _ in scores]
    return [(s - min_score) / (max_score - min_score) for s in scores]

def fill_missing_scores(rankings: List[Tuple[int, float]], num_emails: int) -> List[float]:
    """
    Fills a list of scores with 0.0 where email IDs are missing.

    Args:
        rankings: List of (email_id, score) pairs.
        num_emails: Total number of emails.

    Returns:
        A dense list of scores indexed by email ID - 1.
    """
    filled = [0.0] * num_emails
    for email_id, score in rankings:
        filled[email_id - 1] = score
    return filled

def sigmoid_weight(perplexity: float, midpoint: float = 2500, steepness: float = 0.002) -> float:
    return 1 / (1 + math.exp(steepness * (perplexity - midpoint)))

def get_z_score_top10(score_list):
    score_list_sorted = sorted(score_list, key=lambda x: x[1], reverse=True)
    scores = [email[1] for email in score_list_sorted]
    mean_all = statistics.mean(scores)
    std_all = statistics.stdev(scores)

    top10 = score_list_sorted[:10]
    z_scores_top10 = [(x - mean_all) / std_all if std_all > 0 else 0 for x in top10]
    ave_zscore_top10 = statistics.mean(z_scores_top10)
    return(ave_zscore_top10)

def top10_stand_out(ave_zscore_top10):
    return ave_zscore_top10 >= 2

def get_semantic_weight(query_len: int, query: str, use_perplexity: bool, semantic_score_list, keyword_score_list) -> float:
    semantic_weight = 0.5
    
    if use_perplexity:
        query_perplexity = compute_perplexity(query)
        semantic_weight = sigmoid_weight(query_perplexity)

    if (len(semantic_score_list) > 200) and (len(keyword_score_list) > 200):
        top10_stand_out_semantic = top10_stand_out(get_z_score_top10(semantic_score_list))
        top10_stand_out_keyword = top10_stand_out(get_z_score_top10(keyword_score_list))
        if top10_stand_out_semantic and (not top10_stand_out_keyword):
            semantic_weight += 0.25
        elif (not top10_stand_out_semantic) and top10_stand_out_keyword:
            semantic_weight -= 0.25

    return max(min(semantic_weight, 0.8), 0.2)

def combine_rankings(
    semantic_rankings: List[Tuple[int, float]],
    keyword_rankings: List[Tuple[int, float]],
    query: str,
    query_len: int,
    num_emails: int,
    num_results_wanted: int, 
    is_test=False,
    use_perplexity=False,
) -> List[Tuple[int, float]]:
    """
    Combines semantic and keyword rankings using normalized weighted sum.

    Args:
        semantic_rankings: Semantic search results [(email_id, score)].
        keyword_rankings: Keyword search results [(email_id, score)].
        query_len: Number of words in the query.
        num_emails: Total number of emails.
        num_results_wanted: Number of results to return.
        is_test: If True, return all results sorted by score. If False, return top N results.

    Returns:
        Top N results as list of (email_id, combined_score).
    """
    has_semantic = len(semantic_rankings) > 0
    has_keyword = len(keyword_rankings) > 0

    semantic_scores = fill_missing_scores(semantic_rankings, num_emails) if has_semantic else [0.0] * num_emails
    keyword_scores = fill_missing_scores(keyword_rankings, num_emails) if has_keyword else [0.0] * num_emails

    if has_semantic:
        semantic_scores = min_max_normalize(semantic_scores)
    if has_keyword:
        keyword_scores = min_max_normalize(keyword_scores)

    if has_semantic and has_keyword:
        semantic_weight = get_semantic_weight(query_len, query, use_perplexity, semantic_rankings, keyword_rankings)
    elif has_semantic:
        semantic_weight = 1.0
    elif has_keyword:
        semantic_weight = 0.0
    else:
        # fallback
        return []

    keyword_weight = 1.0 - semantic_weight
    combined_scores = [
        (i + 1, semantic_weight * s + keyword_weight * k)
        for i, (s, k) in enumerate(zip(semantic_scores, keyword_scores))
    ]

    if is_test:
        return sorted(combined_scores, key=lambda x: x[1], reverse=True)
    else:
        return heapq.nlargest(num_results_wanted, combined_scores, key=lambda x: x[1])

def get_top_emails_by_id(top_results: List[Tuple[int, float]], df) -> List[dict]:
    """
    Retrieve emails from dataframe by ID and attach their score.

    Args:
        top_results: List of (email_id, score) pairs.
        df: Pandas DataFrame containing emails.

    Returns:
        List of emails with scores added.
    """
    top_emails = []
    for email_id, score in top_results:
        row = df[df["Id"] == email_id].iloc[0].to_dict()
        row["score"] = score
        top_emails.append(row)
    return top_emails
