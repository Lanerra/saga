# utils/helpers.py
"""Utility functions for the SAGA project."""

from typing import Any

import config


def flatten_dict(
    d: dict[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    """
    Flatten a nested dictionary by concatenating nested keys with a separator.
    Handles nested dictionaries recursively. For lists containing only primitive
    values (non-dict, non-list, non-None), flattens into indexed keys like "key[0]".
    Skips None values and empty lists. Raises ValueError for lists with complex
    (dict or list) items to prevent structure loss.

    Args:
        d: Dictionary to flatten
        parent_key: Prefix to add to keys (used in recursion)
        sep: Separator to use between nested keys

    Returns:
        Flattened dictionary with primitive values only

    Examples:
        >>> flatten_dict({'a': {'b': {'c': 1}}, 'd': 2})
        {'a.b.c': 1, 'd': 2}

        >>> flatten_dict({'key': [1, 2, None, 3]})
        {'key[0]': 1, 'key[1]': 2, 'key[3]': 3}

        >>> flatten_dict({'key': []})
        {}

        >>> flatten_dict({'key': [1, {'nested': 2}]})
        Traceback (most recent call last):
          ...
        ValueError: Cannot flatten complex list at key 'key'

    Raises:
        ValueError: If a list contains non-primitive items (dict or list).
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # Recursively flatten nested dictionaries
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Check if list is primitive (no dict or list items)
            if any(isinstance(item, (dict, list)) for item in v):
                raise ValueError(f"Cannot flatten complex list at key '{new_key}'")
            # Flatten primitive lists with indexed keys, skipping Nones
            for i, item in enumerate(v):
                if item is not None:
                    items.append((f"{new_key}[{i}]", item))
        elif v is not None:
            # Keep primitive values as-is
            items.append((new_key, v))
        # Skip None values to keep the property set clean
    return dict(items)


def _is_fill_in(value: Any) -> bool:
    """Return True if ``value`` is the fill-in placeholder."""
    return isinstance(value, str) and value.strip() == config.FILL_IN
