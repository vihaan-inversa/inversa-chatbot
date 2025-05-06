# main.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from about_me import docs

# Step 1: Generate Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [doc["content"] for doc in docs]

print(texts)
embeddings = model.encode(texts, convert_to_numpy=True)
print(embeddings)
print('embeddings shape: ', embeddings.shape)

# Step 2: Build FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Step 3: Perform a Semantic Search
query = "What are my goals with C++?"
query_embedding = model.encode([query], convert_to_numpy=True)
print('query_embedding shape: ', query_embedding.shape)
print('query_embedding: ', query_embedding)
D, I = index.search(query_embedding, k=3)

print('Results of index.search: ', D, I)

# Step 4: Print Top-k Results
print("\n🔍 Query:", query)
print("\nTop Matches:")
for idx in I[0]:
    print(f"- {docs[idx]['content']} (tag: {docs[idx]['meta']['tag']})")
