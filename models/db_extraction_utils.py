# models/db_extraction_utils.py
"""Utilities for extracting data from Neo4j records and nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import neo4j


class Neo4jExtractor:
    """Utility class for safely extracting data from Neo4j records and nodes."""

    @staticmethod
    def safe_string_extract(value: Any) -> str:
        """Safely extract string from potentially array value."""
        if isinstance(value, list):
            return value[0] if value else ""
        return str(value) if value is not None else ""

    @staticmethod
    def safe_int_extract(value: Any) -> int:
        """Safely extract int from potentially array value."""
        if isinstance(value, list):
            return int(value[0]) if value else 0
        elif isinstance(value, str):
            # Handle comma-separated values by taking first part
            # Clean potential list representation before splitting
            cleaned_value = (
                value.replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .replace('"', "")
            )
            num_str = cleaned_value.split(",")[0].strip()
            return int(num_str) if num_str else 0
        return int(value) if value is not None else 0

    @staticmethod
    def safe_timestamp_extract(value: Any) -> int:
        """Safely extract timestamp from potentially array or comma-separated value."""
        if isinstance(value, list):
            return int(value[0]) if value and value[0] else 0
        elif isinstance(value, str):
            # Handle comma-separated values by taking first part
            cleaned_value = (
                value.replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .replace('"', "")
            )
            timestamp_str = cleaned_value.split(",")[0].strip()
            return int(timestamp_str) if timestamp_str else 0
        return int(value) if value is not None else 0

    @staticmethod
    def safe_list_extract(value: Any) -> list[str]:
        """Safely extract list from potentially mixed value."""
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        return [str(value)] if value is not None else []

    @staticmethod
    def extract_core_fields_from_node(
        node: neo4j.Node | dict[str, Any], core_fields: set[str]
    ) -> dict[str, Any]:
        """Extract additional properties that aren't core fields."""
        return {k: v for k, v in dict(node).items() if k not in core_fields}
