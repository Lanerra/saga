# tests/test_cypher_generation.py
from data_access.cypher_builders.character_cypher import generate_character_node_cypher
from data_access.cypher_builders.native_builders import NativeCypherBuilder
from models import CharacterProfile, WorldItem


def test_generate_character_node_cypher():
    profile = CharacterProfile(name="Alice", description="Hero", traits=["brave"])
    stmts = generate_character_node_cypher(profile)
    assert any("MERGE (c:Character" in s[0] for s in stmts)
    assert any("HAS_CHARACTER" in s[0] for s in stmts)


def test_world_item_upsert_cypher_native():
    item = WorldItem.from_dict(
        "Places",
        "City",
        {"description": "Metropolis", "id": "places_city"},
    )
    cypher, params = NativeCypherBuilder.world_item_upsert_cypher(item, 1)
    assert "MERGE (w:Entity {id: $id})" in cypher
    assert params["id"] == "places_city"


def test_world_item_upsert_cypher_nested_props_native():
    item = WorldItem.from_dict(
        "Places",
        "Echo Forest",
        {"history": {"echo_keepers": "Keepers"}, "id": "places_echo_forest"},
    )
    cypher, params = NativeCypherBuilder.world_item_upsert_cypher(item, 1)
    # Additional props are flattened; content is moved under additional_props
    assert params["id"] == "places_echo_forest"
