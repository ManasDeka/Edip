import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import chromadb
from chromadb.config import Settings 
from typing import List, Dict, Any
from config import (
    CHROMA_HR_COLLECTION,
    CHROMA_IT_COLLECTION,
    CHROMA_FINANCE_COLLECTION,
    CHROMA_OPERATIONS_COLLECTION,
)

# ── ChromaDB Client ─────────────────────────────────────────────────
# PersistentClient auto-creates chroma_db/ folder in project root
_chroma_client = chromadb.PersistentClient(path="./chroma_db")
# _chroma_client = chromadb.PersistentClient(
#     path="./chroma_db",
#     settings=Settings(anonymized_telemetry=False)   # ← Add this line
# )

# ── Collection Map ───────────────────────────────────────────────────
_DOMAIN_COLLECTION_MAP = {
    "HR": CHROMA_HR_COLLECTION,
    "IT": CHROMA_IT_COLLECTION,
    "Finance": CHROMA_FINANCE_COLLECTION,
    "Operations": CHROMA_OPERATIONS_COLLECTION,
}


def _get_or_create_collection(domain: str):
    """Returns the ChromaDB collection for a given domain, creating if not exists."""
    collection_name = _DOMAIN_COLLECTION_MAP.get(domain)
    if not collection_name:
        raise ValueError(f"[ChromaStore] Unknown domain: '{domain}'")
    return _chroma_client.get_or_create_collection(name=collection_name)


def store_chunks(chunks: List[Dict[str, Any]], domain: str) -> None:
    """
    Stores embedded chunks into the appropriate domain-specific ChromaDB collection.

    Args:
        chunks : List of chunk dicts (must include 'embedding' key)
        domain : Classified domain — determines target collection
    """
    collection = _get_or_create_collection(domain)

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        embeddings.append(chunk["embedding"])
        documents.append(chunk["chunk_text"])
        metadatas.append(chunk["metadata"])

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"[ChromaStore] Stored {len(chunks)} chunks → '{_DOMAIN_COLLECTION_MAP[domain]}' collection")
