# utils/common.py
"""Consolidated utilities for SAGA.

This module merges helpers, JSON utilities, YAML parsing, and ingestion tools
into a single import surface to reduce file count and simplify imports.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import yaml

import config

logger = logging.getLogger(__name__)


# --- helpers.py ---
def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary into dot/indexed keys; skip None values."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            if any(isinstance(item, (dict, list)) for item in v):
                raise ValueError(f"Cannot flatten complex list at key '{new_key}'")
            for i, item in enumerate(v):
                if item is not None:
                    items.append((f"{new_key}[{i}]", item))
        elif v is not None:
            items.append((new_key, v))
    return dict(items)


def _is_fill_in(value: Any) -> bool:
    """Return True if value is the fill-in placeholder."""
    return isinstance(value, str) and value.strip() == config.FILL_IN


# --- json_utils.py ---
def extract_json_from_text(text: str) -> str | None:
    """Extract a JSON object/array substring from arbitrary text."""
    if not isinstance(text, str) or not text:
        return None
    first = min([i for i in (text.find("["), text.find("{")) if i != -1] or [len(text)])
    last = max([i for i in (text.rfind("]"), text.rfind("}")) if i != -1] or [-1])
    if first < len(text) and last != -1 and last >= first:
        candidate = text[first : last + 1]
        return candidate if candidate.strip() else None
    return None


def safe_json_loads(
    text: str,
    *,
    expected: type | tuple[type, ...] | None = None,
    strict_extract: bool = False,
) -> Any | None:
    """Safely json.loads text after extraction; optionally validate type."""
    if not isinstance(text, str) or not text.strip():
        return None

    candidate = text
    if strict_extract:
        start_chars = ["{", "["]
        end_chars = {"{": "}", "[": "]"}
        start_pos = min([i for i in (text.find("{"), text.find("[")) if i != -1] or [len(text)])
        if start_pos == len(text):
            return None
        stack: list[str] = []
        end_pos = None
        for idx, ch in enumerate(text[start_pos:], start=start_pos):
            if ch in start_chars:
                stack.append(ch)
            elif ch in ("}", "]"):
                if not stack:
                    return None
                opening = stack.pop()
                if end_chars[opening] != ch:
                    return None
                if not stack:
                    end_pos = idx
                    break
        if end_pos is None:
            return None
        candidate = text[start_pos : end_pos + 1]
    else:
        extracted = extract_json_from_text(text)
        if extracted:
            candidate = extracted

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


# --- yaml_parser.py ---
def normalize_keys_recursive(data: Any) -> Any:
    if isinstance(data, dict):
        new_dict: dict[str, Any] = {}
        for key, value in data.items():
            normalized_key = str(key).lower().replace(" ", "_")
            new_dict[normalized_key] = normalize_keys_recursive(value)
        return new_dict
    if isinstance(data, list):
        return [normalize_keys_recursive(item) for item in data]
    return data


def load_yaml_file(
    filepath: str,
    normalize_keys: bool = True,
    return_none_on_empty: bool = False,
) -> dict[str, Any] | None:
    if not filepath.endswith((".yaml", ".yml")):
        logger.error(f"File specified is not a YAML file: {filepath}")
        return None
    try:
        with open(filepath, encoding="utf-8") as f:
            content = yaml.safe_load(f)
        if content is None:
            return None if return_none_on_empty else {}
        if not isinstance(content, dict):
            logger.error(
                f"YAML file {filepath} must have a dictionary as its root element for this application."
            )
            return None
        return normalize_keys_recursive(content) if normalize_keys else content
    except FileNotFoundError:
        logger.info(f"YAML file '{filepath}' not found (optional).")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {filepath}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while loading YAML file {filepath}: {e}",
            exc_info=True,
        )
        return None


# --- ingestion_utils.py ---
def split_text_into_chapters(
    text: str, max_chars: int = config.MIN_ACCEPTABLE_DRAFT_LENGTH
) -> list[str]:
    """Split text into pseudo-chapters by paragraph boundaries."""
    separator = "\n\n"
    sep_len = len(separator)
    paragraphs = text.split(separator)
    chapters: list[str] = []
    current: list[str] = []
    current_length = 0
    for para in paragraphs:
        para_len = (sep_len + len(para)) if current else len(para)
        if current_length + para_len > max_chars and current:
            chapters.append(separator.join(current).strip())
            current = [para]
            current_length = len(para)
        else:
            current.append(para)
            current_length += para_len
    if current:
        chapters.append(separator.join(current).strip())
    return [c for c in chapters if c.strip()]


__all__ = [
    "flatten_dict",
    "_is_fill_in",
    "extract_json_from_text",
    "safe_json_loads",
    "truncate_for_log",
    "normalize_keys_recursive",
    "load_yaml_file",
    "split_text_into_chapters",
]

