# processing/problem_parser.py
"""Shared utilities for parsing problem lists from LLM JSON output."""

from __future__ import annotations

import json
import logging

from models import ProblemDetail

logger = logging.getLogger(__name__)


def parse_problem_list(text: str, category: str | None = None) -> list[ProblemDetail]:
    """Parse a JSON list of problem details.

    Args:
        text: Raw JSON string from the LLM.
        category: Optional enforced category for each problem.

    Returns:
        A list of ``ProblemDetail`` objects. If the JSON is invalid an error
        entry is returned.
    """
    if not text or not text.strip():
        return []
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if "status" in data and "no significant" in data["status"].lower():
                return []
            if "problems" in data and isinstance(data["problems"], list):
                data = data["problems"]
        if not isinstance(data, list):
            raise ValueError("LLM output was not a list of problems")
    except json.JSONDecodeError as exc:
        logger.error("Failed to decode JSON: %s", exc)
        return [
            {
                "issue_category": category or "meta",
                "problem_description": f"Invalid JSON from LLM: {exc}",
                "quote_from_original_text": "N/A - Invalid JSON",
                "quote_char_start": None,
                "quote_char_end": None,
                "sentence_char_start": None,
                "sentence_char_end": None,
                "suggested_fix_focus": "Ensure LLM outputs valid JSON.",
            }
        ]

    problems: list[ProblemDetail] = []
    for item in data:
        if not isinstance(item, dict):
            logger.warning("Problem item is not a dict: %s", item)
            continue
        prob: ProblemDetail = {
            "issue_category": item.get("issue_category", category or "meta"),
            "problem_description": item.get(
                "problem_description", "N/A - Missing description"
            ),
            "quote_from_original_text": item.get(
                "quote_from_original_text", "N/A - General Issue"
            ),
            "quote_char_start": None,
            "quote_char_end": None,
            "sentence_char_start": None,
            "sentence_char_end": None,
            "suggested_fix_focus": item.get(
                "suggested_fix_focus", "N/A - Missing suggestion"
            ),
        }
        if category:
            prob["issue_category"] = category
        problems.append(prob)
    return problems
