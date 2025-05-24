from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import json

# Load processed chunks
with open("processed_chunks.json", "r", encoding='utf-8') as f:
    chunks = json.load(f)

# Connect to Qdrant
qdrant = QdrantClient(":memory:")  # for local development

# Create collection
collection_name = "travel_chunks"
qdrant.recreate_collection(collection_name, vectors_config={"size": 384, "distance": "Cosine"})

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Upload embeddings
for chunk in chunks:
    vector = model.encode(chunk["text"]).tolist()
    qdrant.upsert(
        collection_name=collection_name,
        points=[{
            "id": chunk["id"],
            "vector": vector,
            "payload": {
                "text": chunk["text"],
                "category": chunk["category"],
                "source": chunk["source"]
            }
        }]
    )

# Sample retrieval
def retrieve(query, top_k=5):
    query_vector = model.encode(query).tolist()
    hits = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    return [hit.payload["text"] for hit in hits]

# Example
print("\n".join(retrieve("What are the best hotels in Cairo?")))