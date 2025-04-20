from src.embeddings.embeddings import EmailEmbedder
from typing import Optional, List, Dict

def semantic_search(query, index, df, top_k) -> List[Dict]:
    print("🧠 Embedding query...")
    embedder = EmailEmbedder(big_model=True)
    query_embedding = embedder.embed_query(query)  # shape: [1, dim]
    query_np = query_embedding.cpu().numpy().astype("float32")

    print("🔍 Running similarity search...")
    #removed top_k stuff because wouldn't really change speed, uses priority queue and needs to score everything anyway
    #k = index.ntotal if top_k is None else min(top_k, index.ntotal) 
    k = index.ntotal
    scores, indices = index.search(query_np, k)

    scores = scores[0]
    indices = indices[0]

    results = []
    #can make more efficient but not that important
    #also could do this outside and avoid passing df around as a parameter
    for idx, score in zip(indices, scores):
        #email_id = int(df.iloc[idx]["Id"])
        email_id = idx + 1 #since not using top_k, and this is consistent wih keyword stuff
        email_info = (email_id, score)
        #email = df.iloc[idx].to_dict()
        #email["score"] = float(score)
        results.append(email_info)

    #sort by Id for efficient combination of rankings with keyword rankings
    results = sorted(results, key=lambda x: x[0]) 
    return results

"""
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
"""
