"""Tests for data_access/cypher_builders/native_builders.py"""

from data_access.cypher_builders.native_builders import NativeCypherBuilder
from models import CharacterProfile, WorldItem


class TestCharacterUpsertCypher:
    """Tests for character upsert Cypher generation."""

    def test_character_upsert_basic(self):
        """Test basic character upsert Cypher."""
        profile = CharacterProfile.from_dict("Alice", {"description": "A hero", "traits": ["brave"]})

        cypher, params = NativeCypherBuilder.character_upsert_cypher(profile, 1)

        assert "MERGE (c:Character {name: $name})" in cypher
        assert params["name"] == "Alice"
        assert params["description"] == "A hero"
        assert params["chapter_number"] == 1

        # Contract: builder-created relationships must be visible to profile reads
        # that filter by r.source_profile_managed.
        assert "source_profile_managed: true" in cypher
        # Contract: traits are updated differentially (do not delete all HAS_TRAIT edges).
        assert "WHERE NOT old_t.name IN $trait_data" in cypher

    def test_character_upsert_with_relationships(self):
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

        assert "MERGE (c:Character {name: $name})" in cypher
        assert params["name"] == "Alice"

    def test_character_upsert_empty_traits(self):
        """Test character upsert with no traits."""
        profile = CharacterProfile.from_dict("Alice", {"description": "A hero", "traits": []})

        cypher, params = NativeCypherBuilder.character_upsert_cypher(profile, 1)

        assert "MERGE (c:Character {name: $name})" in cypher
        assert params["name"] == "Alice"


class TestWorldItemUpsertCypher:
    """Tests for world item upsert Cypher generation."""

    def test_world_item_upsert_basic(self):
        """Test basic world item upsert Cypher."""
        item = WorldItem.from_dict("Locations", "Castle", {"description": "A castle"})

        cypher, params = NativeCypherBuilder.world_item_upsert_cypher(item, 1)

        # P0.2: world upserts must write one canonical world label (derived from category)
        assert "MERGE (w:Location {id: $id})" in cypher
        assert "id" in params
        assert params["name"] == "Castle"
        assert params["category"] == "Locations"

        # Contract: traits are updated differentially (do not delete all HAS_TRAIT edges).
        assert "WHERE NOT old_t.name IN $trait_data" in cypher

        # Contract: relationship targets are no longer forced to :Item; builder supports allowlisted typing.
        assert "apoc.merge.node" in cypher
        assert "world_item_target_label_allowlist" in params

    def test_world_item_upsert_with_goals(self):
        """Test world item upsert with goals."""
        item = WorldItem.from_dict(
            "Locations",
            "Castle",
            {"description": "A castle", "goals": ["Protect the realm"]},
        )

        cypher, params = NativeCypherBuilder.world_item_upsert_cypher(item, 1)

        assert "MERGE (w:Location" in cypher
        assert "goals" in params or "goals" in str(cypher)

    def test_world_item_upsert_with_rules(self):
        """Test world item upsert with rules."""
        item = WorldItem.from_dict("Locations", "Castle", {"description": "A castle", "rules": ["No running"]})

        cypher, params = NativeCypherBuilder.world_item_upsert_cypher(item, 1)

        assert "MERGE (w:Location" in cypher
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

    def test_world_item_upsert_relationship_target_label_and_id(self):
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
