# backend/pinecone_embeds.py
import os
import time
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
region = os.getenv("PINECONE_REGION", "us-east-1")
pc = Pinecone(api_key=api_key)

INDEX_NAME = "pdf-index"  # single index for all PDFs
model = SentenceTransformer("all-MiniLM-L6-v2")

def upsert_embeddings(chunks: list, metadata: dict):
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        vectors.append({
            "id": f"{metadata.get('source')}-{i}",
            "values": embedding,
            "metadata": {**metadata, "chunk_index": i, "text": chunk}
        })

    index = pc.Index(INDEX_NAME)
    
    try:
        stats = index.describe_index_stats()
        if stats.get("total_vector_count", 0) > 0:
            print(f"Clearing index '{INDEX_NAME}' before upserting...")
            index.delete(delete_all=True)
    except Exception as e:
        print(f"Warning: Could not describe/delete index '{INDEX_NAME}': {e}")

    batch_size = 50
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        for attempt in range(3):
            try:
                index.upsert(vectors=batch)
                print(f"Upserted batch {i // batch_size + 1}")
                break
            except Exception as e:
                print(f"Error upserting batch {i // batch_size + 1}, attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)
                if attempt == 2:
                    print("Skipping batch.")
    print(f"Upserted all {len(vectors)} vectors to index '{INDEX_NAME}'.")

def query_pinecone(query_text: str, top_k: int = 500) -> dict:
    index = pc.Index(INDEX_NAME)
    query_vector = model.encode(query_text).tolist()

    try:
        response = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        return {"matches": response.get("matches", [])}
    except Exception as e:
        print(f"Error querying Pinecone index '{INDEX_NAME}': {e}")
        return {"matches": []}

if __name__ == "__main__":
    pass
