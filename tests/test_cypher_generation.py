# tests/test_cypher_generation.py
from data_access.cypher_builders.character_cypher import generate_character_node_cypher
from data_access.cypher_builders.world_cypher import generate_world_element_node_cypher
from models import CharacterProfile, WorldItem


def test_generate_character_node_cypher():
    profile = CharacterProfile(name="Alice", description="Hero", traits=["brave"])
    stmts = generate_character_node_cypher(profile)
    assert any("MERGE (c:Character" in s[0] for s in stmts)
    assert any("HAS_CHARACTER" in s[0] for s in stmts)


def test_generate_world_element_node_cypher():
    item = WorldItem.from_dict(
        "Places",
        "City",
        {"description": "Metropolis"},
    )
    stmts = generate_world_element_node_cypher(item)
    assert any("MERGE (we:Entity" in s[0] for s in stmts)
    merge_stmt = stmts[0]
    assert merge_stmt[1]["id"] == "places_city"
    assert any("CONTAINS_ELEMENT" in s[0] for s in stmts)


def test_generate_world_element_node_cypher_nested_props():
    item = WorldItem.from_dict(
        "Places",
        "Echo Forest",
        {"history": {"echo_keepers": "Keepers"}},
    )
    stmts = generate_world_element_node_cypher(item)
    params = stmts[0][1]
    assert isinstance(params["props"]["history"], str)

    first_stmt, first_params = generate_world_element_node_cypher(item)[0]
    assert "MERGE (we:Entity" in first_stmt
    assert first_params["id"] == "places_echo_forest"
