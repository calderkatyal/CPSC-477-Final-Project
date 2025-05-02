"""
Evaluation metrics for search results.
"""
from typing import List, Dict
import numpy as np
import math
from collections import Counter

def consistency_top_k(ranked_lists, top_k):
    id_counts = Counter()
    for lst in ranked_lists:
        top_k_emails = lst[:top_k]
        id_counts.update(email["Id"] for email in top_k_emails)

    score = sum(math.sqrt(c-1) for c in id_counts.values() if c>1)

    num_lists = len(ranked_lists)
    max_poss_score = top_k * math.sqrt(num_lists - 1) if num_lists > 1 else 1.0
    return score / max_poss_score

def weighted_consistency_top_k(ranked_lists, ks = (10, 20), weights = (0.7, 0.3)):
    scores = [consistency_top_k(ranked_lists, k) for k in ks]
    weighted_score = sum(w*s for w,s in zip(weights, scores))
    return weighted_score

def weighted_kendalls_w(rank_lists: List[List[Dict[str, float]]]) -> float:
    """
    Computes basic Kendall's W coefficient of concordance with print statements.
    Returns 1.0 for perfect agreement, 0.0 for no agreement.
    """
    return 0


def weighted_pairwise_mse(score_lists: List[List[Dict[str, float]]], decay_rate: float = 20.0) -> float:
    """
    Pairwise weighted MSE across K score lists of {Id, score},
    weighting higher-ranked emails more using exponential decay.
    
    Args:
        score_lists: List of K lists of {Id, score}, each representing a query variant.
        decay_rate: Controls how quickly weights drop off for lower-ranked emails.

    Returns:
        Normalized weighted MSE in [0, 1].
    """
    import pdb
    pdb.set_trace()
    K = len(score_lists)
    if K < 2:
        return 0.0

    ref_ids = set(item["Id"] for item in score_lists[0])
    for idx, lst in enumerate(score_lists[1:], start=1):
        assert ref_ids == set(item["Id"] for item in lst), f"ID mismatch in list {idx}"

    email_ids = sorted(ref_ids)

    # Build: email_id → list of (score, rank) for each query
    score_rank_by_id = {email_id: [] for email_id in email_ids}
    for lst in score_lists:
        id_to_score = {item["Id"]: item["score"] for item in lst}
        id_to_rank = {item["Id"]: rank for rank, item in enumerate(lst)}
        for email_id in email_ids:
            score = id_to_score[email_id]
            rank = id_to_rank[email_id]
            score_rank_by_id[email_id].append((score, rank))

    total_weighted_error = 0.0
    total_weight = 0.0

    # For every email, compute pairwise MSE of scores across queries
    for email_id, values in score_rank_by_id.items():
        for i in range(K):
            for j in range(i + 1, K):
                score_i, rank_i = values[i]
                score_j, rank_j = values[j]
                weight = math.exp(-min(rank_i, rank_j) / decay_rate)
                error = (score_i - score_j) ** 2
                total_weighted_error += weight * error
                total_weight += weight

    raw_mse = total_weighted_error / total_weight if total_weight > 0 else 0.0

    return min(1.0, raw_mse / 0.25) 
