import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "/app/data/chroma_db")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

_model = None
_chroma_client = None
_collection = None


def get_embedding_model():
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
    model = get_embedding_model()
    return model.encode([text])[0].tolist()


def store_embedding(doc_id: str, text: str, metadata: dict):
    collection = get_collection()
    embedding = embed_text(text)
    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata]
    )


def search_similar(query_text: str, user_id: str, top_k: int = 5) -> list:
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
    collection = get_collection()
    return collection.count()