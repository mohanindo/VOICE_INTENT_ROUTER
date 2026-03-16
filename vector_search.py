from typing import Any, Dict

import numpy as np

from faiss_index import load_faiss_store
from vector_store import generate_embeddings


def search_faiss(query_text: str, agent_id: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Search matching rules from FAISS by query text.
    Lower score is better because IndexFlatL2 is used.
    """
    if not query_text or not query_text.strip():
        return {"matches": []}

    index, metadata_store = load_faiss_store()

    if index.ntotal == 0:
        return {"matches": []}

    query_embedding = generate_embeddings([query_text])[0]
    query_vector = np.array([query_embedding], dtype="float32")

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
            "score": float(distance),
            "metadata": metadata,
        })

        if len(matches) >= top_k:
            break

    return {"matches": matches}