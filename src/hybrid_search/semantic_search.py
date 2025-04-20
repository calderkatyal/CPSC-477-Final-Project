from src.embeddings.embeddings import EmailEmbedder
from typing import List, Tuple

embedder = EmailEmbedder()

def semantic_search(query: str, index, df) -> List[Tuple[int, float]]:
    """
    Perform semantic search using FAISS over embedded emails.

    Args:
        query: Natural language query.
        index: FAISS index of embeddings.
        df: DataFrame of emails with assigned IDs.

    Returns:
        List of (email_id, similarity_score) sorted by ID.
    """
    print("🧠 Embedding query...")
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
