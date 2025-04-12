from src.utils import load_processed_emails, load_faiss_index, faiss_to_device
from src.embeddings.embeddings import EmailEmbedder
from typing import Optional, List, Dict

def semantic_search(query: str, folder: Optional[str] = None, top_k: Optional[int] = None) -> List[Dict]:
    """
    Perform semantic search on emails using a query string.

    Args:
        query: The search query to embed and compare.
        folder: Optional filter for "inbox" or "sent".
        top_k: Number of top results to return. If None, returns all results.

    Returns:
        A list of dicts, each containing email metadata and similarity score.
    """
    print("🔄 Loading emails and FAISS index...")
    df = load_processed_emails()

    folder = folder.lower() if folder else "inbox"

    if folder not in {"inbox", "sent"}:
        raise ValueError("folder must be 'inbox', 'sent', or None")

    df = df[df["folder"] == folder].reset_index(drop=True)
    index = load_faiss_index(folder)

    assert index.ntotal == len(df), (
        f"FAISS index and DataFrame lengths do not match. "
        f"Length of index: {index.ntotal}, length of DataFrame: {len(df)}"
    )

    print("🧠 Embedding query...")
    embedder = EmailEmbedder(big_model=True)
    query_embedding = embedder.embed_query(query)  # shape: [1, dim]
    query_np = query_embedding.cpu().numpy().astype("float32")

    print("🔍 Running similarity search...")
    k = index.ntotal if top_k is None else min(top_k, index.ntotal)
    scores, indices = index.search(query_np, k)

    scores = scores[0]
    indices = indices[0]

    results = []
    for idx, score in zip(indices, scores):
        email = df.iloc[idx].to_dict()
        email["score"] = float(score)
        results.append(email)

    return results

if __name__ == "__main__":
    query = input("Enter your search query: ")
    results = semantic_search(query, top_k=3)

    for r in results:
        id = r.get("Id")
        subject = r.get("ExtractedSubject") or "No Subject"
        body = r.get("ExtractedBodyText") or "[No Body Content]"
        print(f"\n🔹 Score: {r['score']:.4f}")
        print(f"📧 Email ID: {id}")
        print(f"📌 Subject: {subject[:80]}")
        print(f"✉️ Body Preview: {body[:300]}")
