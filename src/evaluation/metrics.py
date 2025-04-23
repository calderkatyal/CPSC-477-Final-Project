"""
Evaluation metrics for search results.
"""
from typing import List, Dict, Any
import numpy as np
from scipy.stats import kendalltau

def weighted_kendalls_w(rankings: List[List[Dict[str, Any]]]) -> float:
    """Calculate weighted Kendall's W for multiple rankings.

    Args:
        rankings: List of rankings (each ranking is a list of results)

    Returns:
        Weighted Kendall's W score
    """
    # TODO: Implement weighted Kendall's W
    pass

def weighted_mse(scores: List[List[float]]) -> float:
    """Calculate weighted MSE between score lists.

    Args:
        scores: List of score lists

    Returns:
        Weighted MSE score
    """
    # TODO: Implement weighted MSE
    pass
