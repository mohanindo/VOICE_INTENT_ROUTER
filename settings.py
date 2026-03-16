import os
import pickle
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

APP_NAME = "SourceBytes Agent Engine"

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing in environment variables")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
RULE_CACHE_SIZE = int(os.getenv("RULE_CACHE_SIZE", "500"))

FAISS_INDEX_FILE = os.getenv("FAISS_INDEX_FILE", "rule_faiss.index")
FAISS_METADATA_FILE = os.getenv("FAISS_METADATA_FILE", "rule_faiss_metadata.pkl")

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    """
    if not texts:
        return []

    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    return [item.embedding for item in response.data]


def create_faiss_index() -> faiss.IndexFlatL2:
    """
    Create a new FAISS index.
    """
    return faiss.IndexFlatL2(EMBEDDING_DIMENSION)


def load_faiss_store() -> Tuple[faiss.IndexFlatL2, List[Dict[str, Any]]]:
    """
    Load FAISS index and metadata from disk.
    If not present, create empty store.
    """
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_METADATA_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(FAISS_METADATA_FILE, "rb") as f:
            metadata_store = pickle.load(f)
        return index, metadata_store

    return create_faiss_index(), []


def save_faiss_store(index: faiss.IndexFlatL2, metadata_store: List[Dict[str, Any]]) -> None:
    """
    Save FAISS index and metadata to disk.
    """
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(FAISS_METADATA_FILE, "wb") as f:
        pickle.dump(metadata_store, f)


def add_records_to_faiss(records: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
    """
    Add records and embeddings into FAISS store.
    """
    if not records:
        return 0

    if len(records) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(records)} records but {len(embeddings)} embeddings"
        )

    index, metadata_store = load_faiss_store()

    vectors = np.array(embeddings, dtype="float32")
    index.add(vectors)

    for record, embedding in zip(records, embeddings):
        metadata_store.append({
            "text": record["text"],
            "embedding": embedding,
            "metadata": record["metadata"],
        })

    save_faiss_store(index, metadata_store)
    return len(records)


def search_faiss(query_text: str, agent_id: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Search matching rules from FAISS by query text.
    """
    if not query_text or not query_text.strip():
        return {"matches": []}

    index, metadata_store = load_faiss_store()

    if index.ntotal == 0:
        return {"matches": []}

    query_embedding = generate_embeddings([query_text])[0]
    query_vector = np.array([query_embedding], dtype="float32")

    # Search extra records because later we filter by agent_id and duplicate rule_ids
    distances, indices = index.search(query_vector, top_k * 3)

    matches = []
    seen_rule_ids = set()

    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1 or idx >= len(metadata_store):
            continue

        item = metadata_store[idx]
        metadata = item["metadata"]

        if metadata.get("agent_id") != agent_id:
            continue

        rule_id = metadata.get("rule_id")
        if rule_id in seen_rule_ids:
            continue

        seen_rule_ids.add(rule_id)

        matches.append({
            "rule_id": rule_id,
            "score": float(distance),   # lower is better in IndexFlatL2
            "metadata": metadata,
        })

        if len(matches) >= top_k:
            break

    return {"matches": matches}


def rebuild_faiss_store() -> None:
    """
    Delete existing FAISS files.
    """
    if os.path.exists(FAISS_INDEX_FILE):
        os.remove(FAISS_INDEX_FILE)

    if os.path.exists(FAISS_METADATA_FILE):
        os.remove(FAISS_METADATA_FILE)