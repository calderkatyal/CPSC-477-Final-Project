from typing import List, Tuple
from collections import defaultdict

def reciprocal_rank_fusion(results: List[List[Tuple[int, float]]], k: int = 60) -> List[Tuple[int, float]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Args:
        results: A list of ranked lists. Each inner list is a list of (email_id, score) tuples,
                 sorted in descending order of relevance (score).
        k: Constant to dampen the contribution of lower-ranked results. Default is 60.

    Returns:
        A single ranked list of (email_id, fused_score) sorted by fused_score descending.
    """
    fusion_scores = defaultdict(float)

    for result in results:
        for rank, (email_id, _) in enumerate(result):
            fusion_scores[email_id] += 1.0 / (k + rank + 1)

    fused = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
    return fused
