from src.embeddings.embeddings import EmailEmbedder
from src.query_expansion.expander import QueryExpander
from typing import List, Tuple

embedder = None
expander = None

def init_semantic_components(seed=None):
    global embedder, expander
    embedder = EmailEmbedder(seed=seed)
    expander = QueryExpander(seed=seed)

def semantic_search(query: str, index, df) -> List[List[Tuple[int, float]]]:
    """
    Perform semantic search separately for the query and its variants using FAISS.

    Args:
        query: Natural language query.
        index: FAISS index of embeddings.
        df: DataFrame of emails with assigned IDs.

    Returns:
        A list of ranked lists. Each inner list contains (email_id, similarity_score), sorted by similarity score (descending).
    """
    
    assert embedder is not None and expander is not None
    print("🔍 Conducting semantic search...")
   
    print("💡 Generating query variants...")
    queries = expander.expand(query, num_variants=4)
    # print(f"Query variants: {queries}")

    print("🧠 Embedding queries...")
    query_embeddings = embedder.embed_query(queries)

    print("🔍 Searching FAISS index...")

    results_per_variant = []
    for embedding in query_embeddings:
        query_np = embedding.unsqueeze(0).cpu().numpy().astype("float32")
        scores, indices = index.search(query_np, index.ntotal)
        scores, indices = scores[0], indices[0]

        results = []
        for idx, score in zip(indices, scores):
            email_id = int(df.iloc[idx]["Id"])
            results.append((email_id, float(score)))

        results = sorted(results, key=lambda x: x[1], reverse=True)
        results_per_variant.append(results)

    return results_per_variant
