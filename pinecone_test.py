from pinecone import Pinecone, ServerlessSpec
import os

# Connect to Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define your index parameters
index_name = "vihaan-chatbot"
dimension = 1536  # for ada-002
metric = "cosine"

# Check and create if needed
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud="aws",      # or "gcp"
            region="us-east-1"  # or your preferred region (check Pinecone dashboard)
        )
    )
    print(f"✅ Created index '{index_name}'")
else:
    print(f"ℹ️ Index '{index_name}' already exists.")

