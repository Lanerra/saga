# tests/test_db_extraction_utils.py
"""Tests for models/db_extraction_utils.py - Neo4j data extraction utilities."""

import pytest
from unittest.mock import MagicMock

from models.db_extraction_utils import Neo4jExtractor


class TestSafeStringExtract:
    """Tests for safe_string_extract method."""

    def test_safe_string_extract_from_string(self):
        """Test extracting string from string value."""
        result = Neo4jExtractor.safe_string_extract("hello")
        assert result == "hello"

    def test_safe_string_extract_from_int(self):
        """Test extracting string from integer value."""
        result = Neo4jExtractor.safe_string_extract(42)
        assert result == "42"

    def test_safe_string_extract_from_none(self):
        """Test extracting string from None value."""
        result = Neo4jExtractor.safe_string_extract(None)
        assert result == ""

    def test_safe_string_extract_from_empty_list(self):
        """Test extracting string from empty list."""
        result = Neo4jExtractor.safe_string_extract([])
        assert result == ""

    def test_safe_string_extract_from_list_with_value(self):
        """Test extracting string from list with single value."""
        result = Neo4jExtractor.safe_string_extract(["hello"])
        assert result == "hello"

    def test_safe_string_extract_from_list_with_multiple_values(self):
        """Test extracting string from list with multiple values (takes first)."""
        result = Neo4jExtractor.safe_string_extract(["hello", "world"])
        assert result == "hello"

    def test_safe_string_extract_from_list_with_none(self):
        """Test extracting string from list containing None (returns first element)."""
        result = Neo4jExtractor.safe_string_extract([None, "hello"])
        assert result is None  # Returns first element which is None

    def test_safe_string_extract_from_empty_string(self):
        """Test extracting string from empty string."""
        result = Neo4jExtractor.safe_string_extract("")
        assert result == ""

    def test_safe_string_extract_from_bool(self):
        """Test extracting string from boolean value."""
        result = Neo4jExtractor.safe_string_extract(True)
        assert result == "True"

    def test_safe_string_extract_from_float(self):
        """Test extracting string from float value."""
        result = Neo4jExtractor.safe_string_extract(3.14)
        assert result == "3.14"


class TestSafeIntExtract:
    """Tests for safe_int_extract method."""

    def test_safe_int_extract_from_int(self):
        """Test extracting int from integer value."""
        result = Neo4jExtractor.safe_int_extract(42)
        assert result == 42

    def test_safe_int_extract_from_string(self):
        """Test extracting int from string value."""
        result = Neo4jExtractor.safe_int_extract("123")
        assert result == 123

    def test_safe_int_extract_from_none(self):
        """Test extracting int from None value."""
        result = Neo4jExtractor.safe_int_extract(None)
        assert result == 0

    def test_safe_int_extract_from_empty_list(self):
        """Test extracting int from empty list."""
        result = Neo4jExtractor.safe_int_extract([])
        assert result == 0

    def test_safe_int_extract_from_list_with_value(self):
        """Test extracting int from list with single value."""
        result = Neo4jExtractor.safe_int_extract([42])
        assert result == 42

    def test_safe_int_extract_from_list_with_multiple_values(self):
        """Test extracting int from list with multiple values (takes first)."""
        result = Neo4jExtractor.safe_int_extract([100, 200])
        assert result == 100

    def test_safe_int_extract_from_invalid_string(self):
        """Test extracting int from invalid string value."""
        result = Neo4jExtractor.safe_int_extract("not_a_number")
        assert result == 0

    def test_safe_int_extract_from_empty_string(self):
        """Test extracting int from empty string."""
        result = Neo4jExtractor.safe_int_extract("")
        assert result == 0

    def test_safe_int_extract_from_comma_separated_string(self):
        """Test extracting int from comma-separated string (takes first)."""
        result = Neo4jExtractor.safe_int_extract("123,456")
        assert result == 123

    def test_safe_int_extract_from_list_representation_string(self):
        """Test extracting int from string that looks like a list."""
        result = Neo4jExtractor.safe_int_extract("[123]")
        assert result == 123

    def test_safe_int_extract_from_quoted_string(self):
        """Test extracting int from quoted string."""
        result = Neo4jExtractor.safe_int_extract('"123"')
        assert result == 123

    def test_safe_int_extract_from_float(self):
        """Test extracting int from float value (truncates)."""
        result = Neo4jExtractor.safe_int_extract(3.99)
        assert result == 3

    def test_safe_int_extract_from_bool(self):
        """Test extracting int from boolean value."""
        result = Neo4jExtractor.safe_int_extract(True)
        assert result == 1

    def test_safe_int_extract_from_complex_list_representation(self):
        """Test extracting int from complex list string representation."""
        result = Neo4jExtractor.safe_int_extract("['123', '456']")
        assert result == 123


class TestSafeListExtract:
    """Tests for safe_list_extract method."""

    def test_safe_list_extract_from_list(self):
        """Test extracting list from list value."""
        result = Neo4jExtractor.safe_list_extract(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_safe_list_extract_from_none(self):
        """Test extracting list from None value."""
        result = Neo4jExtractor.safe_list_extract(None)
        assert result == []

    def test_safe_list_extract_from_string(self):
        """Test extracting list from string value (wrapped in list)."""
        result = Neo4jExtractor.safe_list_extract("hello")
        assert result == ["hello"]

    def test_safe_list_extract_from_int(self):
        """Test extracting list from integer value (wrapped in list)."""
        result = Neo4jExtractor.safe_list_extract(42)
        assert result == ["42"]

    def test_safe_list_extract_from_empty_list(self):
        """Test extracting list from empty list."""
        result = Neo4jExtractor.safe_list_extract([])
        assert result == []

    def test_safe_list_extract_from_list_with_none_values(self):
        """Test extracting list from list containing None values (filtered out)."""
        result = Neo4jExtractor.safe_list_extract(["a", None, "b", None])
        assert result == ["a", "b"]

    def test_safe_list_extract_from_list_with_mixed_types(self):
        """Test extracting list from list with mixed types."""
        result = Neo4jExtractor.safe_list_extract([1, "hello", None, 3.14])
        assert result == ["1", "hello", "3.14"]

    def test_safe_list_extract_from_bool(self):
        """Test extracting list from boolean value."""
        result = Neo4jExtractor.safe_list_extract(True)
        assert result == ["True"]

    def test_safe_list_extract_from_float(self):
        """Test extracting list from float value."""
        result = Neo4jExtractor.safe_list_extract(3.14)
        assert result == ["3.14"]

    def test_safe_list_extract_from_empty_string(self):
        """Test extracting list from empty string."""
        result = Neo4jExtractor.safe_list_extract("")
        assert result == [""]

    def test_safe_list_extract_from_zero(self):
        """Test extracting list from zero value."""
        result = Neo4jExtractor.safe_list_extract(0)
        assert result == ["0"]


class TestExtractCoreFieldsFromNode:
    """Tests for extract_core_fields_from_node method."""

    def test_extract_core_fields_from_dict(self):
        """Test extracting non-core fields from dictionary node."""
        node = {
            "id": "item-001",
            "name": "Sword",
            "category": "Weapon",
            "description": "Legendary blade",
            "extra_field_1": "value1",
            "extra_field_2": "value2"
        }
        
        core_fields = {"id", "name", "category", "description"}
        
        result = Neo4jExtractor.extract_core_fields_from_node(node, core_fields)
        
        assert len(result) == 2
        assert result["extra_field_1"] == "value1"
        assert result["extra_field_2"] == "value2"
        assert "id" not in result
        assert "name" not in result

    def test_extract_core_fields_from_neo4j_node(self):
        """Test extracting non-core fields from Neo4j node object."""
        # Create a proper dict-like node
        class MockNode:
            def __iter__(self):
                return iter([
                    ("id", "item-002"),
                    ("name", "Shield"),
                    ("category", "Armor"),
                    ("description", "Protective gear"),
                    ("material", "steel"),
                    ("weight", "10kg")
                ])
        
        mock_node = MockNode()
        
        core_fields = {"id", "name", "category", "description"}
        
        result = Neo4jExtractor.extract_core_fields_from_node(mock_node, core_fields)
        
        assert len(result) == 2
        assert result["material"] == "steel"
        assert result["weight"] == "10kg"

    def test_extract_core_fields_empty_result(self):
        """Test extracting when all fields are core."""
        node = {
            "id": "item-003",
            "name": "Helmet",
            "category": "Armor"
        }
        
        core_fields = {"id", "name", "category"}
        
        result = Neo4jExtractor.extract_core_fields_from_node(node, core_fields)
        
        assert len(result) == 0

    def test_extract_core_fields_with_empty_core_set(self):
        """Test extracting when core fields set is empty."""
        node = {
            "id": "item-004",
            "name": "Gauntlet",
            "category": "Armor"
        }
        
        core_fields = set()
        
        result = Neo4jExtractor.extract_core_fields_from_node(node, core_fields)
        
        assert len(result) == 3
        assert result["id"] == "item-004"
        assert result["name"] == "Gauntlet"
        assert result["category"] == "Armor"

    def test_extract_core_fields_with_extra_fields_only(self):
        """Test extracting when node only has extra fields."""
        node = {
            "extra_1": "value1",
            "extra_2": "value2",
            "extra_3": "value3"
        }
        
        core_fields = {"id", "name"}
        
        result = Neo4jExtractor.extract_core_fields_from_node(node, core_fields)
        
        assert len(result) == 3
        assert result["extra_1"] == "value1"
        assert result["extra_2"] == "value2"
        assert result["extra_3"] == "value3"

    def test_extract_core_fields_with_none_values(self):
        """Test extracting when node contains None values."""
        node = {
            "id": "item-005",
            "name": "Boots",
            "extra_field": None
        }
        
        core_fields = {"id", "name"}
        
        result = Neo4jExtractor.extract_core_fields_from_node(node, core_fields)
        
        assert len(result) == 1
        assert result["extra_field"] is None

    def test_extract_core_fields_with_complex_values(self):
        """Test extracting with complex value types."""
        node = {
            "id": "item-006",
            "name": "Cloak",
            "extra_list": [1, 2, 3],
            "extra_dict": {"key": "value"},
            "extra_bool": True
        }
        
        core_fields = {"id", "name"}
        
        result = Neo4jExtractor.extract_core_fields_from_node(node, core_fields)
        
        assert len(result) == 3
        assert result["extra_list"] == [1, 2, 3]
        assert result["extra_dict"] == {"key": "value"}
        assert result["extra_bool"] is True