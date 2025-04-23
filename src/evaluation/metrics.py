"""
Evaluation metrics for search results.
"""
from typing import List, Dict
import numpy as np

def weighted_kendalls_w(score_lists: List[List[Dict[str, float]]]) -> float:
    """
    Compute a normalized weighted Kendall's W (concordance) across K full lists of {Id, score}.
    Higher-ranked emails are given more weight (1 / rank). Returns value in [0, 1].

    Args:
        score_lists: List of K lists of {Id, score}, each representing a query variant.

    Returns:
        Normalized weighted Kendall’s W in [0, 1]. Higher is better.
    """
    K = len(score_lists)
    if K == 0 or any(len(lst) == 0 for lst in score_lists):
        return 0.0

    ref_ids = set(item["Id"] for item in score_lists[0])
    for idx, lst in enumerate(score_lists[1:], start=1):
        assert ref_ids == set(item["Id"] for item in lst), f"ID mismatch in list {idx}"

    email_ids = sorted(ref_ids)
    N = len(email_ids)

    ranks = {email_id: [] for email_id in email_ids}
    weights = {email_id: [] for email_id in email_ids}

    for result_list in score_lists:
        id_to_rank = {item["Id"]: rank for rank, item in enumerate(result_list)}
        for email_id in email_ids:
            r = id_to_rank[email_id]
            w = 1.0 / (r + 1)  # Higher ranks get higher weight
            ranks[email_id].append(r)
            weights[email_id].append(w)

    numerator = 0.0
    denominator = 0.0
    for email_id in email_ids:
        r = np.array(ranks[email_id])
        w = np.array(weights[email_id])
        mean_rank = np.average(r, weights=w)
        numerator += np.dot(w, (r - mean_rank) ** 2)
        denominator += w.sum()

    observed_variance = numerator / denominator if denominator > 0 else 0.0


    ideal_max_ranks = np.arange(N)
    reversed_ranks = np.flip(ideal_max_ranks)
    max_variance = 0.0

    for i in range(K):
        weights = 1.0 / (reversed_ranks + 1)
        mean_rank = np.average(reversed_ranks, weights=weights)
        max_variance += np.dot(weights, (reversed_ranks - mean_rank) ** 2) / weights.sum()

    max_variance /= K

    normalized = 1 - (observed_variance / max_variance) if max_variance > 0 else 1.0
    return max(0.0, min(1.0, normalized))  # Clamp to [0, 1]


def weighted_mse(score_lists: List[List[Dict[str, float]]]) -> float:
    """
    Compute weighted MSE across K full lists of {Id, score},
    weighting higher-ranked emails more (1 / rank).
    Assumes each list has the exact same set of email IDs.

    Args:
        score_lists: List of K lists of {Id, score}, each representing one query variant.

    Returns:
        Weighted MSE (float), normalized to [0, 1].
    """
    K = len(score_lists)
    if K == 0 or any(len(lst) == 0 for lst in score_lists):
        return 0.0


    ref_ids = set(item["Id"] for item in score_lists[0])
    for idx, lst in enumerate(score_lists[1:], start=1):
        assert ref_ids == set(item["Id"] for item in lst), f"ID mismatch in list {idx}"

    id_to_scores = {email_id: [] for email_id in ref_ids}
    id_to_weights = {email_id: [] for email_id in ref_ids}

    for result_list in score_lists:
        for rank, item in enumerate(result_list):
            email_id = item["Id"]
            score = item["score"]
            weight = 1.0 / (rank + 1)
            id_to_scores[email_id].append(score)
            id_to_weights[email_id].append(weight)

    numerator = 0.0
    denominator = 0.0

    for email_id in ref_ids:
        scores = np.array(id_to_scores[email_id])
        weights = np.array(id_to_weights[email_id])
        mean = np.average(scores)
        squared_errors = (scores - mean) ** 2
        numerator += np.dot(weights, squared_errors)
        denominator += weights.sum()

    return numerator / denominator if denominator > 0 else 0.0
