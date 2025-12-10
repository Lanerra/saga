"""Tests for data_access/cypher_builders/native_builders.py"""
from data_access.cypher_builders.native_builders import NativeCypherBuilder
from models import CharacterProfile, WorldItem


class TestCharacterUpsertCypher:
    """Tests for character upsert Cypher generation."""

    def test_character_upsert_basic(self):
        """Test basic character upsert Cypher."""
        profile = CharacterProfile.from_dict(
            "Alice", {"description": "A hero", "traits": ["brave"]}
        )

        cypher, params = NativeCypherBuilder.character_upsert_cypher(profile, 1)

        assert "MERGE (c:Character {name: $name})" in cypher
        assert params["name"] == "Alice"
        assert params["description"] == "A hero"
        assert params["chapter"] == 1

    def test_character_upsert_with_relationships(self):
        """Test character upsert with relationships."""
        profile = CharacterProfile.from_dict(
            "Alice",
            {
                "description": "A hero",
                "traits": ["brave"],
                "relationships": {
                    "Bob": {"type": "FRIEND_OF", "description": "Best friends"}
                },
            },
        )

        cypher, params = NativeCypherBuilder.character_upsert_cypher(profile, 1)

        assert "MERGE (c:Character {name: $name})" in cypher
        assert params["name"] == "Alice"

    def test_character_upsert_empty_traits(self):
        """Test character upsert with no traits."""
        profile = CharacterProfile.from_dict(
            "Alice", {"description": "A hero", "traits": []}
        )

        cypher, params = NativeCypherBuilder.character_upsert_cypher(profile, 1)

        assert "MERGE (c:Character {name: $name})" in cypher
        assert params["name"] == "Alice"


class TestWorldItemUpsertCypher:
    """Tests for world item upsert Cypher generation."""

    def test_world_item_upsert_basic(self):
        """Test basic world item upsert Cypher."""
        item = WorldItem.from_dict("Locations", "Castle", {"description": "A castle"})

        cypher, params = NativeCypherBuilder.world_item_upsert_cypher(item, 1)

        assert "MERGE (w {id: $id})" in cypher or "MERGE (w:Entity {id: $id})" in cypher
        assert "id" in params
        assert params["name"] == "Castle"
        assert params["category"] == "Locations"

    def test_world_item_upsert_with_goals(self):
        """Test world item upsert with goals."""
        item = WorldItem.from_dict(
            "Locations",
            "Castle",
            {"description": "A castle", "goals": ["Protect the realm"]},
        )

        cypher, params = NativeCypherBuilder.world_item_upsert_cypher(item, 1)

        assert "MERGE (w" in cypher
        assert "goals" in params or "goals" in str(cypher)

    def test_world_item_upsert_with_rules(self):
        """Test world item upsert with rules."""
        item = WorldItem.from_dict(
            "Locations", "Castle", {"description": "A castle", "rules": ["No running"]}
        )

        cypher, params = NativeCypherBuilder.world_item_upsert_cypher(item, 1)

        assert "MERGE (w" in cypher
        assert "rules" in params or "rules" in str(cypher)

    def test_world_item_upsert_nested_properties(self):
        """Test world item upsert with nested properties."""
        item = WorldItem.from_dict(
            "Locations",
            "Castle",
            {
                "description": "A castle",
                "history": {"built": "1200", "owner": "King"},
            },
        )

        cypher, params = NativeCypherBuilder.world_item_upsert_cypher(item, 1)

        assert "MERGE (w" in cypher
        assert "id" in params
