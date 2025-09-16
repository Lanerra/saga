# utils/json_utils.py
"""JSON sanitization and safe loading utilities for agent outputs.

Designed for single‑process CLI usage. Avoids heavy dependencies and
centralizes common heuristics used across agents when parsing LLM JSON.
"""

from __future__ import annotations

import json
from typing import Any


def extract_json_from_text(text: str) -> str | None:
    """Extract a JSON object/array substring from arbitrary text.

    Looks for the earliest '[' or '{' and the latest ']' or '}' and
    returns the substring between them if non‑empty.
    """
    if not isinstance(text, str) or not text:
        return None

    first = min([i for i in (text.find("["), text.find("{")) if i != -1] or [len(text)])
    last = max([i for i in (text.rfind("]"), text.rfind("}")) if i != -1] or [-1])
    if first < len(text) and last != -1 and last >= first:
        candidate = text[first : last + 1]
        return candidate if candidate.strip() else None
    return None


def safe_json_loads(
    text: str, *, expected: type | tuple[type, ...] | None = None
) -> Any | None:
    """Safely json.loads text after extraction; optionally validate type.

    Returns None on failure or unexpected type.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    candidate = extract_json_from_text(text) or text
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if expected is not None and not isinstance(obj, expected):
        return None
    return obj


def truncate_for_log(s: str, limit: int = 300) -> str:
    """Return a truncated string for logging purposes."""
    if not isinstance(s, str):
        return ""
    return s if len(s) <= limit else s[:limit] + "..."
