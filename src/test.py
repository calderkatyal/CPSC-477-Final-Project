import pandas as pd
import faiss
from src.config import INBOX_PATH, SENT_PATH, FAISS_INDEX_PATH

def load_parquets():
    print("📦 Loading Parquet files...")
    inbox = pd.read_parquet(INBOX_PATH)
    sent = pd.read_parquet(SENT_PATH)

    print(f"📬 Inbox total: {len(inbox)}")
    print(f"📤 Sent total: {len(sent)}")

    inbox_body_notnull = inbox['ExtractedBodyText'].notnull().sum()
    sent_body_notnull = sent['ExtractedBodyText'].notnull().sum()

    print(f"✅ Inbox w/ body text: {inbox_body_notnull}")
    print(f"✅ Sent w/ body text: {sent_body_notnull}")

    inbox_null_ids = set(inbox[inbox["ExtractedBodyText"].isnull()]["Id"].values)
    sent_null_ids = set(sent[sent["ExtractedBodyText"].isnull()]["Id"].values)

    return inbox, sent, inbox_body_notnull, sent_body_notnull, inbox_null_ids.union(sent_null_ids)

def load_faiss():
    print("\n📈 Loading FAISS index...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"🔢 Index contains: {index.ntotal} vectors")
    return index.ntotal

def compare_lengths(df_length, index_length):
    print(f"\n📊 Comparing counts...")
    if df_length != index_length:
        print(f"❌ Mismatch! DataFrame: {df_length} | FAISS Index: {index_length}")
    else:
        print(f"✅ Match! Both have: {df_length}")

def main():
    inbox, sent, inbox_notnull, sent_notnull, null_id_set = load_parquets()
    total_notnull = inbox_notnull + sent_notnull
    index_count = load_faiss()
    compare_lengths(total_notnull, index_count)

    print("\n🔎 Extra diagnostics:")
    print(f"🟡 Null body text IDs: {len(null_id_set)}")
    print(f"🟡 Total expected (no nulls): {total_notnull}")

    all_ids = pd.concat([inbox, sent])["Id"]
    unique_ids = all_ids.nunique()
    print(f"🆔 Total unique email IDs: {unique_ids}")
    print(f"📋 Total rows: {len(inbox) + len(sent)}")

if __name__ == "__main__":
    main()
