from typing import Any, Dict, List

import numpy as np
from openai import OpenAI

from settings import OPENAI_API_KEY, EMBEDDING_MODEL
from faiss_index import load_faiss_store, save_faiss_store


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
            #"embedding": embedding,
            "metadata": record["metadata"],
        })

    save_faiss_store(index, metadata_store)
    return len(records)


def store_records_in_faiss(records: List[Dict[str, Any]]) -> int:
    """
    Generate embeddings and store the records in FAISS.
    """
    if not records:
        return 0

    texts = [record["text"] for record in records]
    embeddings = generate_embeddings(texts)

    return add_records_to_faiss(records, embeddings)