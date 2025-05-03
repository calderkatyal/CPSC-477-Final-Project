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

def weighted_kendalls_w(score_lists: List[List[Dict[str, float]]], decay_rate: float = 20.0) -> float:
    """
    Computes weighted Kendall's W coefficient measuring ranking agreement.
    Emphasizes agreement on top-ranked items through exponential decay weighting.
    
    Args:
        score_lists: List of K lists of {Id, score}, each representing a ranking
        decay_rate: Controls how quickly weights decrease with rank (higher = slower decay)
        
    Returns:
        Weighted Kendall's W between 0 (no agreement) and 1 (perfect agreement)
    """
    K = len(score_lists)
    if K < 2:
        return 1.0  # Perfect agreement with only one list

    # Get all email IDs from the first list
    email_ids = [item["Id"] for item in score_lists[0]]
    N = len(email_ids)
    if N == 0:
        return 1.0  # No items to rank

    # Build dictionary mapping email ID to its ranks across all systems
    ranks_by_id = {}
    for email_id in email_ids:
        ranks_by_id[email_id] = []
        for lst in score_lists:
            for rank, item in enumerate(lst):
                if item["Id"] == email_id:
                    ranks_by_id[email_id].append(rank)
                    break
    
    # Calculate variance for each item's ranks across systems
    # Higher variance means more disagreement about the item's rank
    variances = {}
    for email_id, ranks in ranks_by_id.items():
        if len(ranks) <= 1:
            variances[email_id] = 0
        else:
            mean_rank = sum(ranks) / len(ranks)
            variances[email_id] = sum((r - mean_rank)**2 for r in ranks) / len(ranks)
    
    # Apply exponential weights based on best (lowest) rank
    weights = {}
    for email_id, ranks in ranks_by_id.items():
        best_rank = min(ranks)
        weights[email_id] = math.exp(-best_rank / decay_rate)
    
    # Calculate weighted average variance
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 1.0  # Edge case - no weights
        
    weighted_avg_variance = sum(weights[email_id] * variances[email_id] 
                               for email_id in email_ids) / total_weight
    
    # Normalize to [0,1] where 1 is perfect agreement
    # Maximum variance for N ranks is (N²-1)/12
    max_variance = (N**2 - 1) / 12 if N > 1 else 1
    
    # Convert to agreement score (1 = perfect agreement, 0 = no agreement)
    agreement = 1.0 - (weighted_avg_variance / max_variance if max_variance > 0 else 0)
    
    return min(1.0, max(0.0, agreement))


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

    K = len(score_lists)
    if K < 2:
        return 0.0

    # print length of each list
    for idx, lst in enumerate(score_lists):
        print(f"List {idx} length: {len(lst)}")
    import pdb
    pdb.set_trace()
    
    """"
    
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

    mse = total_weighted_error / total_weight if total_weight > 0 else 0.0

    return min(1.0, mse ) 
    """
    return 0
