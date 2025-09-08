"""Validation utilities for model fields."""

import re


def _normalize_for_id(text: str) -> str:
    """Normalize a string for use in an ID."""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    # Handle empty string case
    if not text:
        return ""
    # Remove common leading articles to avoid ID duplicates
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"['\"()]", "", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text


def validate_world_item_fields(
    category: str, name: str, item_id: str, allow_empty_name: bool = False
) -> tuple[str, str, str]:
    """Validate and normalize WorldItem core fields, providing defaults for missing values."""
    # Validate category
    if not category or not isinstance(category, str) or not category.strip():
        category = "other"

    # Validate name
    # Only set default name if allow_empty_name is False and name is actually missing/empty
    if (not allow_empty_name) and (
        not name or not isinstance(name, str) or not name.strip()
    ):
        name = "unnamed_element"

    # Validate ID
    if not item_id or not isinstance(item_id, str) or not item_id.strip():
        norm_cat = _normalize_for_id(category)
        norm_name = _normalize_for_id(name)
        # Ensure we have valid normalized IDs
        if not norm_cat:
            norm_cat = "other"
        if not norm_name:
            norm_name = "unnamed"
        item_id = (
            f"{norm_cat}_{norm_name}"
            if norm_cat and norm_name
            else f"element_{hash(category + name)}"
        )

    return category, name, item_id