# tests/test_enhanced_node_taxonomy.py
"""
Comprehensive test suite for enhanced node taxonomy type inference functions.

Tests cover category-based inference, name-based linguistic analysis,
node validation, and classification systems.
"""

import pytest

from core.enhanced_node_taxonomy import (
    validate_node_type,
    get_node_type_priority,
    get_all_node_classifications,
    NodeClassification,
    ENHANCED_NODE_LABELS,
    _is_location_by_structure,
    _is_character_by_structure,
    _is_organization_by_structure,
    _classify_object_by_structure,
    _is_memory_by_structure,
    _is_concept_by_structure,
    _is_event_by_structure,
    _is_system_by_structure,
    _is_energy_by_structure,
)

# Note: The following functions have been removed as part of the migration to unified ML-inspired inference:
# - infer_node_type_from_category (replaced by IntelligentTypeInference)
# - infer_node_type_from_name (replaced by IntelligentTypeInference)
# - suggest_better_node_type (replaced by IntelligentTypeInference)
# Tests for these functions are disabled to avoid breaking the migration.


@pytest.mark.skip(reason="Function infer_node_type_from_category removed in favor of unified ML-inspired inference")
class TestCategoryBasedInference:
    """Test category-based type inference - DISABLED (Function removed)."""
    pass


@pytest.mark.skip(reason="Function infer_node_type_from_name removed in favor of unified ML-inspired inference")
class TestNameBasedInference:
    """Test name-based linguistic type inference - DISABLED (Function removed)."""
    pass


class TestStructuralClassificationHelpers:
    """Test the internal structural classification helper functions."""

    def test_is_location_by_structure(self):
        """Test location detection by structure."""
        location_names = [
            "Dragon Falls",
            "Central Hospital", 
            "Memory Archive",
            "Crystal Nexus",
        ]
        
        non_location_names = [
            "Alice Johnson",
            "Magic Sword",
            "The Battle",
        ]
        
        for name in location_names:
            words = name.split()
            result = _is_location_by_structure(name, words, "")
            assert result, f"Expected {name} to be detected as location"
        
        for name in non_location_names:
            words = name.split()
            result = _is_location_by_structure(name, words, "")
            assert not result, f"Expected {name} to NOT be detected as location"

    def test_is_character_by_structure(self):
        """Test character detection by structure."""
        character_names = [
            "Dr. Smith",
            "Alice Johnson",
            "Captain America",
        ]
        
        non_character_names = [
            "Dragon Falls",
            "Acme Corp",
            "The Battle",
        ]
        
        for name in character_names:
            words = name.split()
            result = _is_character_by_structure(name, words, "")
            assert result, f"Expected {name} to be detected as character"
        
        for name in non_character_names:
            words = name.split()
            result = _is_character_by_structure(name, words, "")
            assert not result, f"Expected {name} to NOT be detected as character"

    def test_is_organization_by_structure(self):
        """Test organization detection by structure."""
        org_names = [
            "Acme Corporation",
            "Tech Inc",
            "The Rebellion",
            "Council of Mages",
        ]
        
        non_org_names = [
            "Alice Johnson",
            "Dragon Falls",
            "Magic Sword",
        ]
        
        for name in org_names:
            words = name.split()
            result = _is_organization_by_structure(name, words, "")
            assert result, f"Expected {name} to be detected as organization"
        
        for name in non_org_names:
            words = name.split()
            result = _is_organization_by_structure(name, words, "")
            assert not result, f"Expected {name} to NOT be detected as organization"

    def test_classify_object_by_structure(self):
        """Test object classification by structure."""
        # Regular objects
        object_names = [
            "Magic Sword",
            "Energy Blaster",
            "Alice's Diary",
        ]
        
        # Artifacts (with special context)
        artifact_context = "ancient legendary magical"
        
        for name in object_names:
            words = name.split()
            result = _classify_object_by_structure(name, words, "")
            assert result == "Object", f"Expected {name} to be classified as Object"
        
        # Test artifact detection
        result = _classify_object_by_structure("Ancient Blade", ["Ancient", "Blade"], artifact_context)
        assert result == "Artifact", f"Expected Ancient Blade with special context to be Artifact"

    def test_is_memory_by_structure(self):
        """Test memory detection by structure."""
        memory_names = [
            "Alice's Memory",
            "Her Childhood",
            "Last Recollection",
        ]
        
        non_memory_names = [
            "Memory Core",  # This is an object
            "Alice Johnson",
            "The Battle",
        ]
        
        for name in memory_names:
            words = name.split()
            result = _is_memory_by_structure(name, words, "")
            assert result, f"Expected {name} to be detected as memory"
        
        for name in non_memory_names:
            words = name.split()
            result = _is_memory_by_structure(name, words, "")
            assert not result, f"Expected {name} to NOT be detected as memory"

    def test_is_concept_by_structure(self):
        """Test concept detection by structure."""
        concept_names = [
            "Democracy",
            "Happiness", 
            "Philosophy",
            "Human Condition",
        ]
        
        non_concept_names = [
            "Alice Johnson",
            "Dragon Falls",
            "Magic Sword",
        ]
        
        for name in concept_names:
            words = name.split()
            result = _is_concept_by_structure(name, words, "")
            assert result, f"Expected {name} to be detected as concept"
        
        for name in non_concept_names:
            words = name.split()
            result = _is_concept_by_structure(name, words, "")
            assert not result, f"Expected {name} to NOT be detected as concept"

    def test_is_event_by_structure(self):
        """Test event detection by structure."""
        event_names = [
            "The Great War",
            "Dragon Battle",
            "Royal Wedding",
            "Murder Mystery",
        ]
        
        non_event_names = [
            "Alice Johnson",
            "Dragon Falls",
            "Magic Sword",
        ]
        
        for name in event_names:
            words = name.split()
            result = _is_event_by_structure(name, words, "")
            assert result, f"Expected {name} to be detected as event"
        
        for name in non_event_names:
            words = name.split()
            result = _is_event_by_structure(name, words, "")
            assert not result, f"Expected {name} to NOT be detected as event"

    def test_is_system_by_structure(self):
        """Test system detection by structure."""
        system_names = [
            "Security System",
            "Data Network",
            "Investigation Protocol",
        ]
        
        non_system_names = [
            "Alice Johnson",
            "Dragon Falls",
            "Magic Sword",
        ]
        
        for name in system_names:
            words = name.split()
            result = _is_system_by_structure(name, words, "")
            assert result, f"Expected {name} to be detected as system"
        
        for name in non_system_names:
            words = name.split()
            result = _is_system_by_structure(name, words, "")
            assert not result, f"Expected {name} to NOT be detected as system"

    def test_is_energy_by_structure(self):
        """Test energy detection by structure."""
        energy_names = [
            "Plasma Energy",
            "Force Field",
            "Power Wave",
        ]
        
        non_energy_names = [
            "Alice Johnson",
            "Dragon Falls",
            "Magic Sword",
        ]
        
        for name in energy_names:
            words = name.split()
            result = _is_energy_by_structure(name, words, "")
            assert result, f"Expected {name} to be detected as energy"
        
        for name in non_energy_names:
            words = name.split()
            result = _is_energy_by_structure(name, words, "")
            assert not result, f"Expected {name} to NOT be detected as energy"


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
            "Entity", "Character", "Location", "Object", "Faction",
            "Event", "Concept", "System", "Energy", "Memory", "Artifact",
            "NovelInfo", "Chapter", "ValueNode", "Trait"
        ]
        
        for node_type in required_types:
            assert node_type in ENHANCED_NODE_LABELS, f"Missing required type: {node_type}"


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
        expected_char = {"SENTIENT", "CONSCIOUS", "PHYSICAL_PRESENCE", "LOCATABLE", "SOCIAL"}
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


@pytest.mark.skip(reason="Function suggest_better_node_type removed in favor of unified ML-inspired inference")
class TestBetterTypeSuggestion:
    """Test type improvement suggestions - DISABLED (Function removed)."""
    pass


@pytest.mark.skip(reason="Functions removed - empty input handling now tested in IntelligentTypeInference tests")
class TestEdgeCases:
    """Test edge cases and error conditions - DISABLED (Functions removed)."""
    pass


class TestContextualInference:
    """Test inference with contextual information."""

    def test_object_context_artifact_promotion(self):
        """Test that special context promotes objects to artifacts."""
        special_contexts = ["ancient", "legendary", "magical", "sacred"]
        
        for context in special_contexts:
            words = ["Ancient", "Sword"]
            result = _classify_object_by_structure("Ancient Sword", words, context)
            assert result == "Artifact", f"Expected Artifact with context '{context}'"

    def test_location_context_detection(self):
        """Test location detection with context clues."""
        context_clues = ["located", "situated", "found", "place"]
        
        for context in context_clues:
            words = ["Unknown", "Place"]
            result = _is_location_by_structure("Unknown Place", words, f"entity {context} here")
            assert result, f"Expected location detection with context '{context}'"


@pytest.mark.skip(reason="Functions removed - disambiguation now handled by IntelligentTypeInference")
class TestRegressionCases:
    """Test specific regression cases and known issues - DISABLED (Functions removed)."""
    pass


@pytest.fixture
def sample_entity_names():
    """Fixture providing sample entity names for testing."""
    return {
        "characters": ["Alice Johnson", "Dr. Smith", "Captain Hook", "Lady Catherine"],
        "locations": ["Dragon Falls", "Central Hospital", "Memory Archive", "Crystal City"],
        "organizations": ["Acme Corp", "Tech Institute", "Workers Union", "The Rebellion"],
        "objects": ["Magic Sword", "Energy Blaster", "Alice's Diary", "Memory Core"],
        "memories": ["Alice's Memory", "Her Childhood", "Last Recollection", "First Day"],
        "concepts": ["Democracy", "Happiness", "Philosophy", "Human Condition"],
        "events": ["The Great War", "Dragon Battle", "Royal Wedding", "Murder Mystery"],
        "systems": ["Security System", "Data Network", "Magic Framework", "Combat Protocol"],
        "energy": ["Plasma Energy", "Force Field", "Power Wave", "Psychic Force"],
    }


@pytest.mark.skip(reason="Functions removed - batch inference now tested in IntelligentTypeInference tests")
class TestBatchInference:
    """Test batch inference operations - DISABLED (Functions removed)."""
    pass