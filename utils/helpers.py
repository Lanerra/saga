"""Utility functions for the SAGA project."""

from typing import Any, Dict

import config


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten a nested dictionary by concatenating nested keys with a separator.
    
    Args:
        d: Dictionary to flatten
        parent_key: Prefix to add to keys (used in recursion)
        sep: Separator to use between nested keys
        
    Returns:
        Flattened dictionary with primitive values only
        
    Example:
        >>> flatten_dict({'a': {'b': {'c': 1}}, 'd': 2})
        {'a.b.c': 1, 'd': 2}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # Recursively flatten nested dictionaries
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to strings to ensure they're primitive
            items.append((new_key, str(v)))
        elif v is not None:
            # Keep primitive values as-is
            items.append((new_key, v))
        # Skip None values to keep the property set clean
    return dict(items)


def is_primitive_type(value: Any) -> bool:
    """
    Check if a value is a primitive type that Neo4j can store as a property.
    
    Args:
        value: Value to check
        
    Returns:
        True if value is a primitive type, False otherwise
    """
    if value is None:
        return True
    primitive_types = (str, int, float, bool)
    return isinstance(value, primitive_types)


def _is_fill_in(value: Any) -> bool:
    """Return True if ``value`` is the fill-in placeholder."""
    return isinstance(value, str) and value.strip() == config.FILL_IN
