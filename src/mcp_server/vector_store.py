import os
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "/app/data/chroma_db")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
DEFAULT_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))

_model = None
_chroma_client = None
_collection = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _chroma_client


def get_collection():
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def embed_text(text: str) -> list:
    """Embed a single text string."""
    model = get_embedding_model()
    return model.encode([text], batch_size=1)[0].tolist()


def embed_texts_batch(texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[list]:
    """
    Embed multiple texts efficiently using batching.
    Processes texts in chunks of batch_size to avoid memory issues
    and improve throughput on large datasets.
    """
    model = get_embedding_model()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, batch_size=batch_size)
        all_embeddings.extend([emb.tolist() for emb in batch_embeddings])
    return all_embeddings


def store_embedding(doc_id: str, text: str, metadata: dict):
    """Store a single text embedding in ChromaDB."""
    collection = get_collection()
    embedding = embed_text(text)
    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata]
    )


def store_embeddings_batch(
    doc_ids: List[str],
    texts: List[str],
    metadatas: List[Dict],
    batch_size: int = DEFAULT_BATCH_SIZE
):
    """
    Store multiple embeddings efficiently using batch processing.
    Ideal for indexing large amounts of historical data.
    Processes in chunks to avoid memory overflow.
    """
    if not doc_ids:
        return
    collection = get_collection()
    embeddings = embed_texts_batch(texts, batch_size=batch_size)
    for i in range(0, len(doc_ids), batch_size):
        collection.upsert(
            ids=doc_ids[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            documents=texts[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )


def search_similar(query_text: str, user_id: str, top_k: int = 5) -> list:
    """Semantic search over ChromaDB filtered by user_id."""
    collection = get_collection()
    query_embedding = embed_text(query_text)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"user_id": user_id} if user_id else None
    )
    output = []
    if results and results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            output.append({
                "content": doc,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "score": 1 - results["distances"][0][i] if results["distances"] else 0.0
            })
    return output


def get_vector_count() -> int:
    """Return total number of vectors stored in ChromaDB."""
    collection = get_collection()
    return collection.count()