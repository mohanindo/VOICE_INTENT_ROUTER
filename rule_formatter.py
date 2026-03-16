"""
rule_formatter.py

Converts parsed Rule objects into vector-ready records for embedding storage,
generates embeddings, stores them in FAISS, and retrieves matching rules.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from rule_model import Rule
from logger import get_logger
from settings import (
    generate_embeddings,
    add_records_to_faiss,
    search_faiss,
    rebuild_faiss_store,
)

logger = get_logger(__name__)

VectorRecord = Dict[str, Any]
Metadata = Dict[str, Any]


# -------------------------------------------------------
# Text Normalization
# -------------------------------------------------------

def normalize_text(text: str | None) -> str:
    """
    Normalize text for embedding consistency.

    Rules:
    - None becomes empty string
    - lowercase
    - strip leading/trailing spaces
    - collapse internal repeated spaces
    """
    if not text:
        return ""

    return " ".join(text.lower().strip().split())


def normalize_list(items: List[str] | None) -> List[str]:
    """
    Normalize a list of strings.
    Removes empty values after normalization.
    """
    if not items:
        return []

    normalized_items: List[str] = []
    for item in items:
        cleaned = normalize_text(item)
        if cleaned:
            normalized_items.append(cleaned)

    return normalized_items


# -------------------------------------------------------
# Metadata Builder
# -------------------------------------------------------

def create_metadata(rule: Rule) -> Metadata:
    """
    Build metadata dictionary stored with vector records.

    This metadata is returned later when the rule is retrieved
    from the vector DB.
    """
    metadata: Metadata = {
        "agent_id": rule.agent_id,
        "rule_id": rule.rule_id,
        "category": rule.category,
        "subcategory": rule.subcategory,
        "sub_subcategory": rule.sub_subcategory,
        "severity": rule.severity,
        "workflow": rule.workflow,
        "escalation": rule.escalation,
        "resolution": rule.resolution,
        "bot_questions": normalize_list(rule.bot_questions),
    }

    return metadata


# -------------------------------------------------------
# Rule -> Vector Records
# -------------------------------------------------------

def convert_rule_to_vector_records(rule: Rule) -> List[VectorRecord]:
    """
    Convert one Rule object into multiple vector records.

    Each intent example becomes one vector record.
    """
    records: List[VectorRecord] = []

    if not rule.intent_examples:
        logger.warning(
            "Skipping rule %s because no intent examples were found",
            rule.rule_id,
        )
        return records

    metadata = create_metadata(rule)

    for example in rule.intent_examples:
        normalized_example = normalize_text(example)

        if not normalized_example:
            continue

        record: VectorRecord = {
            "text": normalized_example,
            "metadata": deepcopy(metadata),
        }
        records.append(record)

    if not records:
        logger.warning(
            "Skipping rule %s because all intent examples were empty after normalization",
            rule.rule_id,
        )

    return records


# -------------------------------------------------------
# Bulk Formatter
# -------------------------------------------------------

def prepare_vector_records(rules: List[Rule]) -> List[VectorRecord]:
    """
    Convert a list of Rule objects into vector records.
    """
    logger.info("Formatting %d rules for embedding", len(rules))

    all_records: List[VectorRecord] = []

    for rule in rules:
        rule_records = convert_rule_to_vector_records(rule)
        all_records.extend(rule_records)

    logger.info("Generated %d vector records", len(all_records))
    return all_records


# -------------------------------------------------------
# Ingestion
# -------------------------------------------------------

def ingest_rules_to_faiss(rules: List[Rule]) -> int:
    """
    Convert rules -> vector records -> embeddings -> FAISS.
    """
    records = prepare_vector_records(rules)

    if not records:
        logger.warning("No vector records found for ingestion")
        return 0

    texts = [record["text"] for record in records]
    embeddings = generate_embeddings(texts)

    stored_count = add_records_to_faiss(records, embeddings)

    logger.info("Stored %d rule records in FAISS", stored_count)
    return stored_count


def rebuild_rules_in_faiss(rules: List[Rule]) -> int:
    """
    Clear old FAISS files and rebuild from scratch.
    """
    rebuild_faiss_store()
    logger.info("Old FAISS store cleared")
    return ingest_rules_to_faiss(rules)


# -------------------------------------------------------
# Retrieval
# -------------------------------------------------------

def search_rules_in_faiss(
    query_text: str,
    agent_id: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Search matching rules for the given user query.
    """
    normalized_query = normalize_text(query_text)
    if not normalized_query:
        return {"matches": []}

    return search_faiss(
        query_text=normalized_query,
        agent_id=agent_id,
        top_k=top_k,
    )


# -------------------------------------------------------
# Local Demo
# -------------------------------------------------------

if __name__ == "__main__":
    from pprint import pprint

    raw_data = [
        {
            "agent_id": "easybuy_support",
            "rule_id": "R12",
            "category": "Billing Issue",
            "subcategory": "Wrong Amount",
            "sub_subcategory": None,
            "intent_examples": [
                "wrong bill",
                "charged extra",
                "billing amount incorrect",
            ],
            "bot_questions": [
                "May I have your bill number?",
            ],
            "severity": "Medium",
            "workflow": "complaint_registration",
            "escalation": "Business Manager",
            "resolution": "Billing will be reviewed by the store team",
            "metadata": {},
        },
        {
            "agent_id": "easybuy_support",
            "rule_id": "R13",
            "category": "Staff Behaviour",
            "subcategory": "Rude Behaviour",
            "sub_subcategory": None,
            "intent_examples": [
                "staff shouted",
                "employee rude",
                "cashier misbehaved",
            ],
            "bot_questions": [
                "Which store did this happen in?",
            ],
            "severity": "Medium",
            "workflow": "complaint_registration",
            "escalation": "Cluster Manager",
            "resolution": "Complaint will be forwarded to store management",
            "metadata": {},
        },
    ]

    try:
        rules = [Rule(**item) for item in raw_data]

        stored_count = rebuild_rules_in_faiss(rules)
        print(f"Stored records in FAISS: {stored_count}")

        result = search_rules_in_faiss(
            query_text="charged extra on my bill",
            agent_id="easybuy_support",
            top_k=3,
        )

        pprint(result)

    except Exception as exc:
        logger.exception("Failed during local rule embedding demo: %s", exc)
        raise