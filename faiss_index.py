import os
import pickle
from typing import Any, Dict, List, Tuple

import faiss

from settings import EMBEDDING_DIMENSION, FAISS_INDEX_FILE, FAISS_METADATA_FILE


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


def rebuild_faiss_store() -> None:
    """
    Delete existing FAISS files.
    """
    if os.path.exists(FAISS_INDEX_FILE):
        os.remove(FAISS_INDEX_FILE)

    if os.path.exists(FAISS_METADATA_FILE):
        os.remove(FAISS_METADATA_FILE)