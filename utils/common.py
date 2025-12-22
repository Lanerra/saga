# utils/common.py
"""Consolidated utilities for SAGA.

This module merges helpers, JSON utilities, YAML parsing, and ingestion tools
into a single import surface to reduce file count and simplify imports.
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog
import yaml

import config

logger = structlog.get_logger(__name__)

_JSON_FENCE_PATTERN = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?```",
    flags=re.DOTALL | re.IGNORECASE,
)


# --- helpers.py ---
def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary into dot/indexed keys; skip None values."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            if any(isinstance(item, dict | list) for item in v):
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


def _extract_balanced_json_substring(text: str) -> str | None:
    if not isinstance(text, str) or not text:
        return None

    start_chars = ["{", "["]
    end_chars = {"{": "}", "[": "]"}
    start_pos = min([i for i in (text.find("{"), text.find("[")) if i != -1] or [len(text)])
    if start_pos == len(text):
        return None

    stack: list[str] = []
    end_pos: int | None = None
    for idx, ch in enumerate(text[start_pos:], start=start_pos):
        if ch in start_chars:
            stack.append(ch)
            continue

        if ch in ("}", "]"):
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
    return candidate if candidate.strip() else None


def extract_json_candidates_from_response(response: str) -> list[tuple[str, str]]:
    """
    Extract candidate JSON strings from a possibly-chatty LLM response.

    Supports:
    - Pure JSON (object or list)
    - JSON in markdown fences ```json ... ```
    - Embedded valid JSON preceded/followed by commentary (via JSONDecoder.raw_decode)
    - Balanced-bracket substring salvage (conservative)

    Returns a list of (source, json_string) candidates, ordered from most likely
    to least likely, de-duplicated while preserving order.
    """
    text = (response or "").strip()
    if not text:
        return []

    candidates: list[tuple[str, str]] = [("raw", text)]

    for i, match in enumerate(_JSON_FENCE_PATTERN.finditer(response or ""), start=1):
        block = (match.group(1) or "").strip()
        if block:
            candidates.append((f"fence[{i}]", block))

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", response or ""):
        index = match.start()
        try:
            _obj, end = decoder.raw_decode(response or "", index)
        except Exception:
            continue
        snippet = (response or "")[index:end].strip()
        if snippet:
            candidates.append((f"raw_decode@{index}", snippet))

    balanced = _extract_balanced_json_substring(response or "")
    if balanced:
        candidates.append(("balanced_substring", balanced))

    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for source, candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append((source, candidate))

    return unique


def try_load_json_from_response(
    response: str,
    *,
    expected_root: type | tuple[type, ...] | None = None,
    wrapper_keys: tuple[str, ...] = (),
) -> tuple[Any | None, list[tuple[str, str]], list[str]]:
    """
    Try to parse JSON from an LLM response using multiple extraction strategies.

    Returns:
        (parsed_or_none, candidates_tried, parse_errors)

    Notes:
        - If wrapper_keys is provided and the parsed value is an object containing exactly
          one of those keys, the value at that key is unwrapped.
        - If expected_root is provided, parsed values that don't match are rejected.
    """
    candidates = extract_json_candidates_from_response(response)
    if not candidates:
        return None, [], ["empty response"]

    parse_errors: list[str] = []
    for source, candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as error:
            parse_errors.append(f"{source}: JSONDecodeError at pos {error.pos}: {error.msg}")
            continue

        if wrapper_keys and isinstance(parsed, dict):
            for wrapper_key in wrapper_keys:
                if wrapper_key in parsed:
                    parsed = parsed[wrapper_key]
                    break

        if expected_root is not None and not isinstance(parsed, expected_root):
            parse_errors.append(
                f"{source}: wrong root type {type(parsed).__name__} (expected {expected_root})"
            )
            continue

        return parsed, candidates, parse_errors

    return None, candidates, parse_errors


def load_json_with_contract_then_salvage(*, raw_text: str, expected_root: type) -> Any:
    """
    Strict JSON-only contract loader with salvage behavior.

    Behavior:
    - First attempts json.loads(raw_text).
    - On JSONDecodeError, tries to salvage embedded JSON using
      [`extract_json_candidates_from_response()`](utils/common.py:1) parsing.
    - If salvage fails, re-raises the original JSONDecodeError.

    This matches the contract behavior used by initialization parsing code.
    """
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as original_error:
        salvaged, _candidates, _errors = try_load_json_from_response(
            raw_text,
            expected_root=expected_root,
        )
        if salvaged is None:
            raise original_error
        return salvaged


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
        extracted_strict = _extract_balanced_json_substring(text)
        if extracted_strict is None:
            return None
        candidate = extracted_strict
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


def ensure_exact_keys(*, value: Any, required_keys: set[str], context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object, got {type(value).__name__}")

    value_keys = {str(key) for key in value.keys()}
    if value_keys != required_keys:
        raise ValueError(
            f"{context} must contain exactly keys {sorted(required_keys)} (got {sorted(value_keys)})"
        )


def ensure_required_keys(*, value: Any, required_keys: set[str], context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object, got {type(value).__name__}")

    value_keys = {str(key) for key in value.keys()}
    missing = sorted(required_keys - value_keys)
    if missing:
        raise ValueError(f"{context} is missing required keys {missing}")


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
            logger.error(f"YAML file {filepath} must have a dictionary as its root element for this application.")
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
def split_text_into_chapters(text: str, max_chars: int = 8000) -> list[str]:
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
    "extract_json_candidates_from_response",
    "try_load_json_from_response",
    "load_json_with_contract_then_salvage",
    "safe_json_loads",
    "ensure_exact_keys",
    "ensure_required_keys",
    "truncate_for_log",
    "normalize_keys_recursive",
    "load_yaml_file",
    "split_text_into_chapters",
]
