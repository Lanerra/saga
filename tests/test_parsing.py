# tests/test_parsing.py
from agents.knowledge_agent import (  # Assuming parsing is now a module inside kg_maintainer
    parse_unified_character_updates,
    parse_unified_world_updates,
)
from models import (
    CharacterProfile,
    WorldItem,
)  # Added WorldItem import

# Example of a helper to create JSON strings for tests if they get complex
# def make_json_string(data_dict):
#     return json.dumps(data_dict)


def test_parse_character_updates_simple_json():
    json_text = """
    {
        "Alice": {
            "description": "Brave hero from JSON",
            "traits": ["smart", "resourceful"],
            "status": "Adventuring",
            "relationships": {
                "Bob": "Ally"
            },
            "development_in_chapter_1": "Embarked on a quest."
        },
        "Bob": {
            "description": "Loyal companion",
            "traits": ["strong", "kind"],
            "status": "Assisting Alice"
        }
    }
    """
    result = parse_unified_character_updates(json_text, 1)
    assert "Alice" in result
    alice_prof = result["Alice"]
    assert isinstance(alice_prof, CharacterProfile)
    assert alice_prof.name == "Alice"
    assert alice_prof.description == "Brave hero from JSON"
    assert "smart" in alice_prof.traits
    assert "resourceful" in alice_prof.traits
    assert alice_prof.status == "Adventuring"
    assert alice_prof.relationships.get("Bob") == "Ally"
    assert alice_prof.updates["development_in_chapter_1"] == "Embarked on a quest."

    assert "Bob" in result
    bob_prof = result["Bob"]
    assert isinstance(bob_prof, CharacterProfile)
    assert bob_prof.description == "Loyal companion"
    # Test default development note if not provided by JSON but other attrs exist
    assert (
        bob_prof.updates["development_in_chapter_1"]
        == "Character 'Bob' details updated in Chapter 1."
    )


def test_parse_character_updates_empty_or_invalid_json():
    assert parse_unified_character_updates("", 1) == {}
    assert parse_unified_character_updates("{}", 1) == {}
    assert parse_unified_character_updates("[]", 1) == {}  # Not a dict of characters
    assert parse_unified_character_updates("invalid json", 1) == {}
    # Test with a valid JSON but not the expected structure (e.g. list at root)
    assert parse_unified_character_updates('[{"name": "Alice"}]', 1) == {}


def test_parse_world_updates_simple_json():
    json_text = """
    {
        "Locations": {
            "City of Brightness": {
                "description": "A large, well-lit city from JSON.",
                "atmosphere": "Vibrant and bustling"
            },
            "Dark Forest": {
                "description": "A mysterious and old forest.",
                "atmosphere": "Eerie and quiet",
                "features": ["Ancient ruins", "Hidden paths"]
            }
        },
        "Factions": {
            "The Sun Guild": {
                "description": "A guild that worships the sun.",
                "goals": ["Spread light", "Help others"],
                "elaboration_in_chapter_1": "Introduced as a benevolent force."
            }
        },
        "Overview": {
            "description": "A world of magic and mystery.",
            "mood": "Adventurous"
        }
    }
    """
    result = parse_unified_world_updates(json_text, 1)
    assert isinstance(result, dict)

    assert "Locations" in result
    locations = result["Locations"]
    assert "City of Brightness" in locations
    city = locations["City of Brightness"]
    assert isinstance(city, WorldItem)
    assert city.name == "City of Brightness"
    assert city.category == "Locations"
    assert city.description == "A large, well-lit city from JSON."
    assert city.additional_properties["atmosphere"] == "Vibrant and bustling"
    # Check for default elaboration note
    assert (
        city.additional_properties["elaboration_in_chapter_1"]
        == "Item 'City of Brightness' in category 'Locations' updated in Chapter 1."
    )

    assert "Dark Forest" in locations
    forest = locations["Dark Forest"]
    assert isinstance(forest, WorldItem)
    assert forest.description == "A mysterious and old forest."
    assert "Ancient ruins" in forest.additional_properties["features"]

    assert "Factions" in result
    factions = result["Factions"]
    assert "The Sun Guild" in factions
    guild = factions["The Sun Guild"]
    assert isinstance(guild, WorldItem)
    assert "Spread light" in guild.goals
    assert (
        guild.additional_properties["elaboration_in_chapter_1"]
        == "Introduced as a benevolent force."
    )  # Explicitly provided

    assert "Overview" in result
    overview_cat = result["Overview"]
    assert "_overview_" in overview_cat  # Overview item is stored with key "_overview_"
    overview_item = overview_cat["_overview_"]
    assert isinstance(overview_item, WorldItem)
    assert overview_item.category == "Overview"  # Category name from JSON
    assert overview_item.name == "_overview_"  # Fixed name for overview item
    assert overview_item.description == "A world of magic and mystery."
    assert overview_item.additional_properties["mood"] == "Adventurous"
    assert (
        overview_item.additional_properties["elaboration_in_chapter_1"]
        == "Overall world overview updated in Chapter 1."
    )


def test_parse_world_updates_empty_or_invalid_json():
    assert parse_unified_world_updates("", 1) == {}
    assert parse_unified_world_updates("{}", 1) == {}
    assert parse_unified_world_updates("[]", 1) == {}  # Not a dict of categories
    assert parse_unified_world_updates("invalid json", 1) == {}
    # Test with valid JSON but not expected structure (e.g. category content is not a dict of items)
    assert parse_unified_world_updates('{"Locations": ["City1", "City2"]}', 1) == {}


# It might be good to add more tests:
# - Character updates with relationships in different formats (list of strings, list of dicts)
# - World updates with list items for details (e.g. rules, key_elements)
# - Normalization of keys if LLM provides slightly different casings/spacings for keys
# - Handling of completely unexpected structures in the JSON from LLM
# - Test for the default elaboration notes more thoroughly.
