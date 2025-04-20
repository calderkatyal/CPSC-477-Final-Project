from src.embeddings.embeddings import EmailEmbedder
from typing import List, Tuple
import torch

def semantic_search(query: str, index, df) -> List[Tuple[int, float]]:
    """
    Perform semantic search on emails using a query string.

    Args:
        query: The search query.
        index: Pre-loaded FAISS index for the folder.
        df: Corresponding DataFrame (must contain 'Id' column).

    Returns:
        A list of (email_id, similarity score) tuples, sorted by ID.
    """
    print("🧠 Embedding query...")
    embedder = EmailEmbedder(big_model=True)
    query_embedding = embedder.embed_query(query)
    query_np = query_embedding.cpu().numpy().astype("float32")

    print("🔍 Running similarity search...")
    scores, indices = index.search(query_np, index.ntotal)
    scores, indices = scores[0], indices[0]

    results = []
    for idx, score in zip(indices, scores):
        email_id = int(df.iloc[idx]["Id"])
        results.append((email_id, float(score)))

    return sorted(results, key=lambda x: x[0])