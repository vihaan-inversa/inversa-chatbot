from openai import OpenAI
from pinecone import Pinecone
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("vihaan-chatbot")

MODEL = "text-embedding-ada-002"

def embed_query(text):
    res = client.embeddings.create(
        model=MODEL,
        input=[text]
    )
    return res.data[0].embedding

def semantic_search(query, k=5):
    query_vec = embed_query(query)
    res = index.query(
        vector=query_vec,
        top_k=k,
        include_metadata=True
    )
    return res["matches"]

if __name__ == "__main__":
    query = input("🔍 Ask something: ").strip()
    results = semantic_search(query)

    print(f"\n🔍 Query: {query}\n")
    for i, match in enumerate(results):
        score = f"{match['score']:.3f}"
        meta  = match['metadata']
        source = meta.get("source", meta.get("tag", "unknown"))
        preview = meta["content"][:150].replace("\n", " ") + "..."
        print(f"{i+1}. ({score}) from [{source}]:\n   {preview}\n")
