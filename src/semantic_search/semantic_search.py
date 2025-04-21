from src.embeddings.embeddings import EmailEmbedder
from src.query_expansion.expander import QueryExpander
from typing import List, Tuple

embedder = EmailEmbedder()
expander = QueryExpander()

def semantic_search(query: str, index, df) -> List[Tuple[int, float]]:
    """
    Generate embeddings for the query and its variants, then perform a similarity search using FAISS over email embeddings.

    Args:
        query: Natural language query.
        index: FAISS index of embeddings.
        df: DataFrame of emails with assigned IDs.

    Returns:
        List of (email_id, similarity_score) sorted by ID.
    """
    print("🧠 Generating query variants...")
    queries = expander.expand(query, num_variants=3) + [query]
    print(f"Query variants: {queries}")
    print("🧠 Embedding query and its variants...")
    query_embeddings = embedder.embed_query(queries)
    query_embedding = query_embeddings.mean(dim=0, keepdim=True) # TODO: Change this to RRF and also use variants in keyword search
    query_np = query_embedding.cpu().numpy().astype("float32")

    print("🔍 Running similarity search...")
    scores, indices = index.search(query_np, index.ntotal)
    scores, indices = scores[0], indices[0]

    results = []
    for idx, score in zip(indices, scores):
        email_id = int(df.iloc[idx]["Id"])
        results.append((email_id, float(score)))

    return sorted(results, key=lambda x: x[0])
