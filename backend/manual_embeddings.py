# backend/manual_embeddings.py
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Global in-memory store for manual embeddings
manual_store = []

def upsert_embeddings(chunks: list, metadata: dict):
    """
    Generate embeddings for each chunk and store them in a global list.
    This clears any previous entries.
    """
    global manual_store
    manual_store = []  # Clear previous embeddings
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk)
        manual_store.append({
            "id": f"{metadata.get('source')}-{i}",
            "embedding": embedding,
            "text": chunk,
            "metadata": {**metadata, "chunk_index": i}
        })
    print(f"[Manual] Upserted {len(chunks)} chunks.")

def query_manual(query_text: str, top_k: int = 5) -> dict:
    """
    Encode the query, compute cosine similarities against stored embeddings,
    and return the top_k matches in a dictionary format.
    """
    query_embedding = model.encode(query_text)
    similarities = []
    for item in manual_store:
        sim = cosine_similarity([query_embedding], [item["embedding"]])[0][0]
        similarities.append((sim, item))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    # Get top_k matches and format results
    matches = []
    for sim, match in similarities[:top_k]:
        matches.append({
            "id": match["id"],
            "metadata": match["metadata"],
            "text": match["text"],
            "distance": 1 - sim  # Lower distance means more similar
        })
    return {"matches": matches}
