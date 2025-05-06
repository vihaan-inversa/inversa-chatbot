import os
import openai
from pinecone import Pinecone
from about_me import docs  # your list of {"content": str, "meta": {...}}

openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("vihaan-chatbot")

MODEL = "text-embedding-ada-002"

# --- 1. Embed function ---
def embed_texts(texts):
    res = openai.Embedding.create(
        input=texts,
        model=MODEL
    )
    return [d["embedding"] for d in res["data"]]

# --- 2. Upload to Pinecone ---
batch_size = 50
for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
    texts = [d["content"] for d in batch]
    embeddings = embed_texts(texts)
    
    to_upsert = []
    for j, doc in enumerate(batch):
        idx = str(i + j)
        vec = embeddings[j]
        meta = doc.get("meta", {})
        meta["content"] = doc["content"]  # include text itself in metadata
        to_upsert.append((idx, vec, meta))
    
    index.upsert(vectors=to_upsert)
    print(f"✅ Uploaded batch {i}–{i+len(batch)-1}")

print("🎉 All docs uploaded.")
