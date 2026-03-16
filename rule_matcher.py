"""
rule_matcher.py

Selects the best rule from FAISS search results
and converts it into structured decision output.
"""

from typing import Dict, Any, Optional


def match_rule(search_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract best rule from FAISS search result
    and convert it into structured rule output.
    """

    matches = search_result.get("matches", [])

    if not matches:
        return None

    best_rule = matches[0]
    metadata = best_rule.get("metadata", {})
    bot_questions = metadata.get("bot_questions", [])

    return {
        "rule_id": metadata.get("rule_id"),
        "category": metadata.get("category"),
        "subcategory": metadata.get("subcategory"),
        "severity": metadata.get("severity"),
        "workflow": metadata.get("workflow"),
        "escalation": metadata.get("escalation"),
        "score": best_rule.get("score"),
        "bot_question": bot_questions[0] if bot_questions else None,
        
    }