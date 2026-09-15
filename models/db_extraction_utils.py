# models/db_extraction_utils.py
"""Extract typed values from Neo4j nodes and query results.

This module provides small, intentionally permissive helpers for normalizing values
returned from Neo4j queries. The driver and some query patterns may yield values in
unexpected shapes (for example, lists containing a single value).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import neo4j


class Neo4jExtractor:
    """Extract normalized values from Neo4j record/node values.

    Notes:
        These helpers prefer producing a usable default over raising. Callers that
        require strict validation should validate upstream before calling.
    """

    @staticmethod
    def safe_string_extract(value: Any) -> str:
        """Normalize a value into a string.

        Args:
            value: Value returned from a Neo4j record or node property.

        Returns:
            A string representation of the value. If `value` is a list, the first element
            is used. If `value` is `None`, an empty string is returned.
        """
        if isinstance(value, list):
            return value[0] if value else ""
        return str(value) if value is not None else ""

    @staticmethod
    def safe_int_extract(value: Any) -> int:
        """Normalize a value into an integer.

        Args:
            value: Value returned from a Neo4j record or node property.

        Returns:
            An integer parsed from the input. If `value` is a list, the first element is
            used. If `value` is a string containing multiple comma-separated values, the
            first segment is used. If parsing fails, `0` is returned.
        """
        if isinstance(value, list):
            return int(value[0]) if value else 0
        elif isinstance(value, str):
            # Handle comma-separated values by taking first part
            # Clean potential list representation before splitting
            cleaned_value = value.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
            num_str = cleaned_value.split(",")[0].strip()
            if not num_str:
                return 0
            try:
                return int(num_str)
            except (ValueError, TypeError):
                return 0
        try:
            return int(value) if value is not None else 0
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def safe_list_extract(value: Any) -> list[str]:
        """Normalize a value into a list of strings.

        Args:
            value: Value returned from a Neo4j record or node property.

        Returns:
            A list of strings. If `value` is a list, `None` entries are removed and the
            remaining entries are stringified. If `value` is a scalar, a single-item list
            is returned. If `value` is `None`, an empty list is returned.
        """
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        return [str(value)] if value is not None else []

    @staticmethod
    def extract_core_fields_from_node(node: neo4j.Node | dict[str, Any], core_fields: set[str]) -> dict[str, Any]:
        """Extract non-core properties from a node-like mapping.

        Args:
            node: Neo4j node or dictionary of node properties.
            core_fields: Property names considered part of the canonical model.

        Returns:
            A dictionary containing properties not present in `core_fields`.
        """
        return {k: v for k, v in dict(node).items() if k not in core_fields}
