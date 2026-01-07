# tests/test_kg_models.py
"""Tests for models/kg_models.py - core knowledge graph data models."""

from unittest.mock import MagicMock

import pytest

from models.kg_models import CharacterProfile, RelationshipUsage, WorldItem


class TestCharacterProfile:
    """Tests for CharacterProfile model and factory methods."""

    def test_from_dict_basic(self):
        """Test creating character from dictionary with known fields."""
        data = {"description": "Protagonist", "traits": ["brave", "intelligent"], "status": "Active", "created_chapter": 1, "is_provisional": False}

        result = CharacterProfile.from_dict("Alice", data)

        assert result.name == "Alice"
        assert result.description == "Protagonist"
        assert result.traits == ["brave", "intelligent"]
        assert result.status == "Active"
        assert result.created_chapter == 1
        assert result.is_provisional is False
        assert result.updates == {}

    def test_from_dict_with_extra_fields(self):
        """Test that extra fields are stored in updates."""
        data = {"description": "Side character", "age": 30, "occupation": "Farmer", "extra_field": "value"}

        result = CharacterProfile.from_dict("Bob", data)

        assert result.name == "Bob"
        assert result.description == "Side character"
        # Extra fields are accessible via updates dict
        assert result.updates["age"] == 30
        assert result.updates["occupation"] == "Farmer"
        assert result.updates["extra_field"] == "value"
        assert "age" in result.updates
        assert "occupation" in result.updates
        assert "extra_field" in result.updates

    def test_from_dict_with_updates_field(self):
        """Test merging updates field with extra fields."""
        data = {"description": "Mysterious", "updates": {"age": 25, "secret": "hidden"}, "extra_field": "value"}

        result = CharacterProfile.from_dict("Charlie", data)

        assert result.name == "Charlie"
        # Extra fields are accessible via updates dict
        assert result.updates["age"] == 25
        assert result.updates["secret"] == "hidden"
        assert result.updates["extra_field"] == "value"
        assert result.updates == {"age": 25, "secret": "hidden", "extra_field": "value"}

    def test_to_dict(self):
        """Test converting character to flat dictionary."""
        profile = CharacterProfile(
            name="Diana", description="Hero", traits=["strong", "compassionate"], status="Active", created_chapter=2, is_provisional=False, updates={"age": 28, "power_level": "high"}
        )

        result = profile.to_dict()

        assert result["description"] == "Hero"
        assert result["traits"] == ["strong", "compassionate"]
        assert result["status"] == "Active"
        assert result["created_chapter"] == 2
        assert result["is_provisional"] is False
        assert result["age"] == 28
        assert result["power_level"] == "high"
        assert "name" not in result  # name should be excluded

    def test_from_dict_record_with_relationships(self):
        """Test creating character from query record with relationships."""
        record = {
            "c": {"name": "Eve", "description": "Spy", "status": "Active", "created_chapter": 3, "is_provisional": False},
            "relationships": [{"target_name": "Alice", "type": "FRIENDS_WITH", "description": "Childhood friends"}, {"target_name": "Bob", "type": "WORKS_WITH", "description": "Colleagues"}],
            "traits": ["stealthy", "observant"],
        }

        result = CharacterProfile.from_dict_record(record)

        assert result.name == "Eve"
        assert result.description == "Spy"
        assert result.traits == ["stealthy", "observant"]
        assert result.status == "Active"
        assert len(result.relationships) == 2
        assert "Alice" in result.relationships
        assert result.relationships["Alice"]["type"] == "FRIENDS_WITH"
        assert result.relationships["Bob"]["description"] == "Colleagues"

    def test_from_dict_record_without_relationships(self):
        """Test creating character from query record without relationships."""
        record = {"c": {"name": "Frank", "description": "Lone wolf", "status": "Unknown", "created_chapter": 0, "is_provisional": True}}

        result = CharacterProfile.from_dict_record(record)

        assert result.name == "Frank"
        assert result.description == "Lone wolf"
        assert result.relationships == {}
        assert result.is_provisional is True

    def test_from_dict_record_with_traits_in_node(self):
        """Test creating character with traits stored in node property."""
        record = {"c": {"name": "Grace", "description": "Leader", "traits": ["charismatic", "decisive"], "status": "Active"}}

        result = CharacterProfile.from_dict_record(record)

        assert result.name == "Grace"
        assert result.traits == ["charismatic", "decisive"]

    def test_from_db_record(self):
        """Test creating character from Neo4j record."""
        # Mock Neo4j record
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: {"c": {"name": "Heidi", "description": "Explorer", "status": "Active"}}[key]

        result = CharacterProfile.from_db_record(mock_record)

        assert result.name == "Heidi"
        assert result.description == "Explorer"

    def test_from_db_node_with_dict(self):
        """Test creating character from dictionary node."""
        node_dict = {"name": "Ivan", "description": "Scholar", "traits": ["wise", "patient"], "status": "Active", "created_chapter": 4, "is_provisional": False}

        result = CharacterProfile.from_db_node(node_dict)

        assert result.name == "Ivan"
        assert result.description == "Scholar"
        assert result.traits == ["wise", "patient"]
        assert result.relationships == {}  # Should be empty for from_db_node

    def test_from_db_node_with_neo4j_node(self):
        """Test creating character from Neo4j node object."""

        # Create a proper dict-like object that behaves like a Neo4j node
        class MockNode:
            def __iter__(self):
                return iter([("name", "Judy"), ("description", "Detective"), ("traits", ["observant", "logical"]), ("status", "Active")])

        mock_node = MockNode()

        result = CharacterProfile.from_db_node(mock_node)

        assert result.name == "Judy"
        assert result.description == "Detective"
        assert result.traits == ["observant", "logical"]

    def test_to_cypher_params(self):
        """Test building Cypher parameter dictionary."""
        profile = CharacterProfile(name="Kevin", description="Engineer", traits=["technical", "creative"], status="Active", created_chapter=5, is_provisional=False, updates={"age": 35})

        result = profile.to_cypher_params()

        assert result["name"] == "Kevin"
        assert result["description"] == "Engineer"
        assert result["traits"] == ["technical", "creative"]
        assert result["status"] == "Active"
        assert result["created_chapter"] == 5
        assert result["is_provisional"] is False
        # relationships and updates should be excluded
        assert "relationships" not in result
        assert "updates" not in result


class TestWorldItem:
    """Tests for WorldItem model and factory methods."""

    def test_from_dict_basic(self):
        """Test creating world item from dictionary."""
        data = {
            "id": "item-001",
            "category": "Weapon",
            "name": "Excalibur",
            "description": "Legendary sword",
            "goals": ["defeat evil"],
            "rules": ["only wielded by true king"],
            "key_elements": ["shining blade", "ancient runes"],
            "traits": ["magical", "powerful"],
            "created_chapter": 1,
            "is_provisional": False,
        }

        result = WorldItem.from_dict("Weapon", "Excalibur", data)

        assert result.id == "item-001"
        assert result.category == "Weapon"
        assert result.name == "Excalibur"
        assert result.description == "Legendary sword"
        assert result.goals == ["defeat evil"]
        assert result.rules == ["only wielded by true king"]
        assert result.key_elements == ["shining blade", "ancient runes"]
        assert result.traits == ["magical", "powerful"]
        assert result.created_chapter == 1
        assert result.is_provisional is False

    def test_from_dict_with_extra_properties(self):
        """Test that extra fields are stored in additional_properties."""
        data = {"id": "item-002", "description": "King's castle", "population": 5000, "era": "medieval"}

        result = WorldItem.from_dict("Location", "Camelot", data)

        assert result.name == "Camelot"
        # Extra fields are accessible via additional_properties dict
        assert result.additional_properties["population"] == 5000
        assert result.additional_properties["era"] == "medieval"
        assert "population" in result.additional_properties
        assert "era" in result.additional_properties

    def test_to_dict(self):
        """Test converting world item to flat dictionary."""
        item = WorldItem(
            id="item-003",
            category="Artifact",
            name="Holy Grail",
            description="Sacred cup",
            goals=["bring peace"],
            rules=["only found by pure heart"],
            key_elements=["golden", "radiant"],
            traits=["magical"],
            created_chapter=2,
            is_provisional=False,
            additional_properties={"material": "gold", "origin": "heaven"},
        )

        result = item.to_dict()

        assert result["description"] == "Sacred cup"
        assert result["goals"] == ["bring peace"]
        assert result["material"] == "gold"
        assert result["origin"] == "heaven"
        # id, category, name should be excluded
        assert "id" not in result
        assert "category" not in result
        assert "name" not in result

    def test_from_dict_record_with_relationships(self):
        """Test creating world item from query record with relationships."""
        record = {
            "w": {"id": "item-004", "category": "Creature", "name": "Dragon", "description": "Mythical beast", "created_chapter": 3, "is_provisional": False},
            "relationships": [{"target_name": "Excalibur", "type": "GUARDS", "description": "protects treasure"}, {"target_name": "Camelot", "type": "LIVES_NEAR", "description": "in mountains"}],
            "traits": ["fire-breathing", "ancient"],
        }

        result = WorldItem.from_dict_record(record)

        assert result.id == "item-004"
        assert result.category == "Creature"
        assert result.name == "Dragon"
        assert result.traits == ["fire-breathing", "ancient"]
        assert len(result.relationships) == 2
        assert "Excalibur" in result.relationships

    def test_from_dict_record_without_node(self):
        """Test error when record doesn't contain world element node."""
        record = {"other": "data"}

        with pytest.raises(ValueError, match="No world element node found in record"):
            WorldItem.from_dict_record(record)

    def test_from_dict_record_with_we_alias(self):
        """Test creating world item from record with 'we' alias."""
        record = {"we": {"id": "item-005", "category": "Magic", "name": "Spellbook", "description": "Ancient tome"}}

        result = WorldItem.from_dict_record(record)

        assert result.id == "item-005"
        assert result.category == "Magic"
        assert result.name == "Spellbook"

    def test_from_db_record(self):
        """Test creating world item from Neo4j record."""

        # Create a proper dict-like record
        class MockRecord:
            def __getitem__(self, key):
                return {"w": {"id": "item-006", "category": "Location", "name": "Forest", "description": "Enchanted woods"}}[key]

            def get(self, key, default=None):
                try:
                    return self[key]
                except KeyError:
                    return default

        mock_record = MockRecord()

        result = WorldItem.from_db_record(mock_record)

        assert result.id == "item-006"
        assert result.category == "Location"

    def test_from_db_node_with_dict(self):
        """Test creating world item from dictionary node."""
        node_dict = {
            "id": "item-007",
            "category": "Object",
            "name": "Mirror",
            "description": "Magic mirror",
            "goals": ["show truth"],
            "traits": ["magical", "sentient"],
            "created_chapter": 4,
            "is_provisional": False,
            "extra_prop": "value",
        }

        result = WorldItem.from_db_node(node_dict)

        assert result.id == "item-007"
        assert result.category == "Object"
        assert result.name == "Mirror"
        # extra_prop should be in additional_properties
        assert "extra_prop" in result.additional_properties

    def test_from_db_node_with_neo4j_node(self):
        """Test creating world item from Neo4j node object."""

        # Create a proper dict-like object that behaves like a Neo4j node
        class MockNode:
            def __iter__(self):
                return iter([("id", "item-008"), ("category", "Creature"), ("name", "Phoenix"), ("description", "Fire bird"), ("traits", ["immortal", "powerful"]), ("created_chapter", 5)])

        mock_node = MockNode()

        result = WorldItem.from_db_node(mock_node)

        assert result.id == "item-008"
        assert result.category == "Creature"
        assert result.name == "Phoenix"

    def test_to_cypher_params(self):
        """Test building Cypher parameter dictionary for world item."""
        item = WorldItem(
            id="item-009",
            category="Artifact",
            name="Amulet",
            description="Protective charm",
            goals=["ward off evil"],
            rules=["must be worn"],
            key_elements=["gold", "gemstone"],
            traits=["magical"],
            created_chapter=6,
            is_provisional=False,
            additional_properties={"weight": "10g"},
        )

        result = item.to_cypher_params()

        assert result["id"] == "item-009"
        assert result["name"] == "Amulet"
        assert result["category"] == "Artifact"
        assert result["goals"] == ["ward off evil"]
        assert result["additional_props"]["weight"] == "10g"
        # relationships should be excluded
        assert "relationships" not in result


class TestRelationshipUsage:
    """Tests for RelationshipUsage dataclass."""

    def test_from_dict_basic(self):
        """Test creating relationship usage from dictionary."""
        data = {
            "canonical_type": "FRIENDS_WITH",
            "first_used_chapter": 1,
            "usage_count": 5,
            "example_descriptions": ["childhood friends", "close companions"],
            "synonyms": ["CLOSE_FRIENDS", "BEST_FRIENDS"],
            "last_used_chapter": 10,
        }

        result = RelationshipUsage.from_dict(data)

        assert result.canonical_type == "FRIENDS_WITH"
        assert result.first_used_chapter == 1
        assert result.usage_count == 5
        assert result.example_descriptions == ["childhood friends", "close companions"]
        assert result.synonyms == ["CLOSE_FRIENDS", "BEST_FRIENDS"]
        assert result.last_used_chapter == 10

    def test_from_dict_with_missing_optional_fields(self):
        """Test creating relationship usage with missing optional fields."""
        data = {"canonical_type": "WORKS_WITH", "first_used_chapter": 2, "usage_count": 3}

        result = RelationshipUsage.from_dict(data)

        assert result.canonical_type == "WORKS_WITH"
        assert result.first_used_chapter == 2
        assert result.usage_count == 3
        assert result.example_descriptions == []
        assert result.synonyms == []
        assert result.last_used_chapter == 0
        assert result.embedding is None

    def test_to_dict(self):
        """Test converting relationship usage to dictionary."""
        usage = RelationshipUsage(
            canonical_type="LOVES", first_used_chapter=1, usage_count=2, example_descriptions=["romantic love"], embedding=[0.1, 0.2, 0.3], synonyms=["IN_LOVE_WITH"], last_used_chapter=5
        )

        result = usage.to_dict()

        assert result["canonical_type"] == "LOVES"
        assert result["first_used_chapter"] == 1
        assert result["usage_count"] == 2
        assert result["example_descriptions"] == ["romantic love"]
        assert result["embedding"] == [0.1, 0.2, 0.3]
        assert result["synonyms"] == ["IN_LOVE_WITH"]
        assert result["last_used_chapter"] == 5

    def test_relationship_usage_equality(self):
        """Test that relationship usage instances can be compared."""
        usage1 = RelationshipUsage(canonical_type="TEST", first_used_chapter=1, usage_count=1)

        usage2 = RelationshipUsage(canonical_type="TEST", first_used_chapter=1, usage_count=1)

        assert usage1 == usage2

        # Modify one field
        usage2.usage_count = 2
        assert usage1 != usage2
