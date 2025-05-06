from openai import OpenAI
from pinecone import Pinecone
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("vihaan-chatbot")

EMBED_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4"  # or "gpt-3.5-turbo" if needed

def embed_query(text):
    res = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text]
    )
    return res.data[0].embedding

def semantic_search(query, k=5):
    query_vec = embed_query(query)
    res = index.query(vector=query_vec, top_k=k, include_metadata=True)
    return [match['metadata']['content'] for match in res['matches']]

def generate_answer(query, contexts):
    context_text = "\n\n".join(contexts)
    prompt = f"""You are Vihaan, an AI assistant answering as yourself.

Use the context below to answer the question. If the context is insufficient, say so.

Context:
\"\"\"
{context_text}
\"\"\"

Q: {query}
A:"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    query = input("🔍 Ask something: ").strip()
    top_chunks = semantic_search(query)
    answer = generate_answer(query, top_chunks)
    print(f"\n🤖 Answer:\n{answer}\n")
