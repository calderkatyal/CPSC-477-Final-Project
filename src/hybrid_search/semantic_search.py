from src.utils import load_processed_emails, load_faiss_index
from src.embeddings.embeddings import EmailEmbedder
from typing import Optional, List, Tuple

def semantic_search(query: str, index, df) -> List[Tuple[int, float]]:
    """
    Perform semantic search on emails using a query string.

    Args:
        query: The search query to embed and compare.
        index: Pre-loaded FAISS index
        df: DataFrame of emails corresponding to the FAISS index (must contain 'Id')

    Returns:
        A list of (email_id, similarity score) tuples.
    """
    print("🧠 Embedding query...")
    embedder = EmailEmbedder(big_model=True)
    query_embedding = embedder.embed_query(query)  # shape: [1, dim]
    query_np = query_embedding.cpu().numpy().astype("float32")

    print("🔍 Running similarity search...")
    k = index.ntotal  # no top_k — filter after combining
    scores, indices = index.search(query_np, k)

    scores = scores[0]
    indices = indices[0]

    results = []
    for idx, score in zip(indices, scores):
        email_id = int(df.iloc[idx]["Id"])
        results.append((email_id, float(score)))

    # Optional: sort by ID for alignment with keyword search
    return sorted(results, key=lambda x: x[0])
