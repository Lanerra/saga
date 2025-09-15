# tests/test_enhanced_node_taxonomy.py
"""
Comprehensive test suite for enhanced node taxonomy type inference functions.

Tests cover category-based inference, name-based linguistic analysis,
node validation, and classification systems.
"""

import pytest

from core.enhanced_node_taxonomy import (
    ENHANCED_NODE_LABELS,
    NodeClassification,
    get_all_node_classifications,
    get_node_type_priority,
    validate_node_type,
)


class TestNodeValidation:
    """Test node type validation."""

    def test_validate_node_type_valid(self):
        """Test validation of valid node types."""
        valid_types = [
            "Character",
            "Location",
            "Object",
            "Faction",
            "Event",
            "Concept",
            "System",
            "Energy",
            "Memory",
            "Artifact",
        ]

        for node_type in valid_types:
            result = validate_node_type(node_type)
            assert result, f"Expected {node_type} to be valid"

    def test_validate_node_type_invalid(self):
        """Test validation of invalid node types."""
        invalid_types = [
            "InvalidType",
            "NotAType",
            "FakeType",
            "",
            None,
        ]

        for node_type in invalid_types:
            result = validate_node_type(node_type)
            assert not result, f"Expected {node_type} to be invalid"

    def test_enhanced_node_labels_completeness(self):
        """Test that ENHANCED_NODE_LABELS contains expected types."""
        required_types = [
            "Entity",
            "Character",
            "Location",
            "Object",
            "Faction",
            "Event",
            "Concept",
            "System",
            "Energy",
            "Memory",
            "Artifact",
            "NovelInfo",
            "Chapter",
            "ValueNode",
            "Trait",
        ]

        for node_type in required_types:
            assert (
                node_type in ENHANCED_NODE_LABELS
            ), f"Missing required type: {node_type}"


class TestNodeClassifications:
    """Test node classification system."""

    def test_sentient_classification(self):
        """Test SENTIENT classification."""
        sentient_types = ["Character", "Person", "Deity", "Spirit"]
        non_sentient_types = ["Object", "Location", "Concept", "System"]

        for node_type in sentient_types:
            assert node_type in NodeClassification.SENTIENT

        for node_type in non_sentient_types:
            assert node_type not in NodeClassification.SENTIENT

    def test_conscious_classification(self):
        """Test CONSCIOUS classification."""
        conscious_types = ["Character", "Person", "Deity"]
        non_conscious_types = ["Object", "Location", "Spirit", "Concept"]

        for node_type in conscious_types:
            assert node_type in NodeClassification.CONSCIOUS

        for node_type in non_conscious_types:
            assert node_type not in NodeClassification.CONSCIOUS

    def test_physical_presence_classification(self):
        """Test PHYSICAL_PRESENCE classification."""
        physical_types = ["Character", "Object", "Location", "Structure"]
        non_physical_types = ["Concept", "Memory", "Trait", "ValueNode"]

        for node_type in physical_types:
            assert node_type in NodeClassification.PHYSICAL_PRESENCE

        for node_type in non_physical_types:
            assert node_type not in NodeClassification.PHYSICAL_PRESENCE

    def test_ownable_classification(self):
        """Test OWNABLE classification."""
        ownable_types = ["Object", "Artifact", "Currency", "Structure"]
        non_ownable_types = ["Character", "Person", "Concept", "Memory"]

        for node_type in ownable_types:
            assert node_type in NodeClassification.OWNABLE

        for node_type in non_ownable_types:
            assert node_type not in NodeClassification.OWNABLE

    def test_get_all_node_classifications(self):
        """Test getting all classifications for a node type."""
        # Test Character (should have multiple classifications)
        char_classifications = get_all_node_classifications("Character")
        expected_char = {
            "SENTIENT",
            "CONSCIOUS",
            "PHYSICAL_PRESENCE",
            "LOCATABLE",
            "SOCIAL",
        }
        assert expected_char.issubset(char_classifications)

        # Test Object (should have different classifications)
        obj_classifications = get_all_node_classifications("Object")
        expected_obj = {"PHYSICAL_PRESENCE", "LOCATABLE", "OWNABLE"}
        assert expected_obj.issubset(obj_classifications)

        # Test Concept (should be abstract)
        concept_classifications = get_all_node_classifications("Concept")
        assert "ABSTRACT" in concept_classifications
        assert "PHYSICAL_PRESENCE" not in concept_classifications


class TestNodeTypePriority:
    """Test node type priority system."""

    def test_get_node_type_priority(self):
        """Test priority calculation for node types."""
        # More specific types should have higher priority
        character_priority = get_node_type_priority("Character")
        entity_priority = get_node_type_priority("Entity")

        assert character_priority > entity_priority

        # Test some specific types
        artifact_priority = get_node_type_priority("Artifact")
        object_priority = get_node_type_priority("Object")

        assert artifact_priority > object_priority  # Artifact is more specific

    def test_priority_for_unknown_type(self):
        """Test priority for unknown/invalid types."""
        unknown_priority = get_node_type_priority("UnknownType")
        assert unknown_priority == 0


@pytest.fixture
def sample_entity_names():
    """Fixture providing sample entity names for testing."""
    return {
        "characters": ["Alice Johnson", "Dr. Smith", "Captain Hook", "Lady Catherine"],
        "locations": [
            "Dragon Falls",
            "Central Hospital",
            "Memory Archive",
            "Crystal City",
        ],
        "organizations": [
            "Acme Corp",
            "Tech Institute",
            "Workers Union",
            "The Rebellion",
        ],
        "objects": ["Magic Sword", "Energy Blaster", "Alice's Diary", "Memory Core"],
        "memories": [
            "Alice's Memory",
            "Her Childhood",
            "Last Recollection",
            "First Day",
        ],
        "concepts": ["Democracy", "Happiness", "Philosophy", "Human Condition"],
        "events": ["The Great War", "Dragon Battle", "Royal Wedding", "Murder Mystery"],
        "systems": [
            "Security System",
            "Data Network",
            "Magic Framework",
            "Combat Protocol",
        ],
        "energy": ["Plasma Energy", "Force Field", "Power Wave", "Psychic Force"],
    }
