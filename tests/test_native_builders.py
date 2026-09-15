# tests/test_native_builders.py
"""Tests for data_access/cypher_builders/native_builders.py"""

from data_access.cypher_builders.native_builders import NativeCypherBuilder
from models import CharacterProfile, WorldItem


class TestCharacterUpsertCypher:
    """Tests for character upsert Cypher generation."""

    def test_character_upsert_basic(self) -> None:
        """Test basic character upsert Cypher."""
        profile = CharacterProfile.from_dict("Alice", {"description": "A hero", "traits": ["brave"], "relationships": {"Bob": {"type": "FRIEND_OF", "description": "A declared friend"}}})

        cypher, params = NativeCypherBuilder.character_upsert_cypher(profile, 1)

        assert "WITH 'Character' AS entity_label, $name AS entity_name, $id AS supplied_id" in cypher
        assert params["name"] == "Alice"
        assert params["description"] == "A hero"
        assert params["chapter_number"] == 1

        # Contract: builder-created relationships must be visible to profile reads
        # that filter by r.source_profile_managed.
        assert len(params["relationship_data"]) == 1
        assert params["relationship_data"][0]["properties"]["source_profile_managed"] is True
        assert "rel_data.properties" in cypher
        # Contract: traits are now stored as node properties
        assert "SET c.traits = $trait_data" in cypher

    def test_character_upsert_with_relationships(self) -> None:
        """Test character upsert with relationships."""
        profile = CharacterProfile.from_dict(
            "Alice",
            {
                "description": "A hero",
                "traits": ["brave"],
                "relationships": {"Bob": {"type": "FRIEND_OF", "description": "Best friends"}},
            },
        )

        cypher, params = NativeCypherBuilder.character_upsert_cypher(profile, 1)

        assert "WITH 'Character' AS entity_label, $name AS entity_name, $id AS supplied_id" in cypher
        assert params["name"] == "Alice"

    def test_character_upsert_empty_traits(self) -> None:
        """Test character upsert with no traits."""
        profile = CharacterProfile.from_dict("Alice", {"description": "A hero", "traits": []})

        cypher, params = NativeCypherBuilder.character_upsert_cypher(profile, 1)

        assert "WITH 'Character' AS entity_label, $name AS entity_name, $id AS supplied_id" in cypher
        assert params["name"] == "Alice"


class TestWorldItemUpsertCypher:
    """Tests for world item upsert Cypher generation."""

    def test_world_item_upsert_basic(self) -> None:
        """Test basic world item upsert Cypher."""
        item = WorldItem.from_dict("Locations", "Castle", {"description": "A castle"})

        cypher, params = NativeCypherBuilder.world_item_upsert_cypher(item, 1)

        assert "RETURN node AS w" in cypher
        assert params["primary_label"] == "Location"
        assert "id" in params
        assert params["name"] == "Castle"
        assert params["category"] == "Locations"

        # Contract: traits are now stored as node properties
        assert "SET w.traits = $trait_data" in cypher

        # Contract: relationship targets are no longer forced to :Item; builder supports allowlisted typing.
        assert "apoc.merge.node" in cypher
        assert "world_item_target_label_allowlist" in params

    def test_world_item_upsert_with_goals(self) -> None:
        """Test world item upsert with goals."""
        item = WorldItem.from_dict(
            "Locations",
            "Castle",
            {"description": "A castle", "goals": ["Protect the realm"]},
        )

        cypher, params = NativeCypherBuilder.world_item_upsert_cypher(item, 1)

        assert "RETURN node AS w" in cypher
        assert params["primary_label"] == "Location"
        assert params["goals"] == ["Protect the realm"]

    def test_world_item_upsert_with_rules(self) -> None:
        """Test world item upsert with rules."""
        item = WorldItem.from_dict("Locations", "Castle", {"description": "A castle", "rules": ["No running"]})

        cypher, params = NativeCypherBuilder.world_item_upsert_cypher(item, 1)

        assert "RETURN node AS w" in cypher
        assert params["primary_label"] == "Location"
        assert params["rules"] == ["No running"]

    def test_world_item_upsert_nested_properties(self) -> None:
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

        assert "RETURN node AS w" in cypher
        assert "id" in params

    def test_world_item_upsert_relationship_target_label_and_id(self) -> None:
        """World relationship targets can optionally specify target_label + target_id (allowlisted)."""
        item = WorldItem.from_dict(
            "Locations",
            "Castle",
            {
                "description": "A castle",
                "relationships": {
                    "Gate": {
                        "type": "LEADS_TO",
                        "description": "Exit gate to the outer ward",
                        "target_label": "Location",
                        "target_id": "locations_gate",
                    }
                },
            },
        )

        cypher, params = NativeCypherBuilder.world_item_upsert_cypher(item, 1)

        assert "apoc.merge.node" in cypher
        assert "world_item_target_label_allowlist" in params
        assert isinstance(params.get("relationship_data"), list)
        assert params["relationship_data"][0]["target_name"] == "Gate"
        assert params["relationship_data"][0]["target_label"] == "Location"
        assert params["relationship_data"][0]["target_id"] == "locations_gate"
