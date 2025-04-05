# backend/chromadb_embeds.py
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_storage")
COLLECTION_NAME = "pdf_collection"  # single collection for all PDFs

def upsert_embeddings(chunks: list, metadata: dict):
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
    
    # Clear the collection by fetching all IDs and deleting them
    existing = collection.get()
    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])
    
    documents = []
    ids = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        ids.append(f"{metadata.get('source')}-{i}")
        metadatas.append({**metadata, "chunk_index": i})

    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    print(f"[ChromaDB] Upserted {len(chunks)} chunks to {COLLECTION_NAME}")

def query_chroma(query: str, top_k=1000):
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
    max_allowed = 100
    total_records = len(collection.get()["ids"])
    actual_top_k = min(top_k, total_records, max_allowed)

    if actual_top_k == 0:
        return {"matches": []}

    results = collection.query(
        query_texts=[query],
        n_results=actual_top_k,
        include=["documents", "metadatas", "distances"]
    )

    matches = []
    for i in range(len(results["ids"][0])):
        matches.append({
            "id": results["ids"][0][i],
            "metadata": results["metadatas"][0][i],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i]
        })

    return {"matches": matches}
