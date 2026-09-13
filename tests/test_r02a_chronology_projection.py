"""R07-13 chronological extraction and R07-15 lossless profile projection."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.commit_entity_conversion import _convert_to_character_profiles, _convert_to_world_items
from core.langgraph.nodes.scene_extraction_normalization import consolidate_scene_extractions
from core.langgraph.state import ExtractedEntity
from data_access import character_queries
from data_access.cypher_builders.native_builders import NativeCypherBuilder
from models.kg_models import CharacterProfile, WorldItem
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager
from tests.test_r02a_named_candidates import SCENE, responses, run_extraction, selected_scene


@pytest.mark.parametrize("kind,collection", [("Character", "characters"), ("Location", "world_items"), ("Event", "world_items")])
def test_later_shorter_assertion_wins_with_lossless_history(kind: str, collection: str) -> None:
    first = {"name": "Ada", "type": kind, "description": "A long earlier description stating the old situation.", "first_appearance_chapter": 1, "scene_index": 0,
             "attributes": {"id": "exact-17", "status": "Alive", "category": kind, "traits": ["brave"], "key_elements": ["Old fact"], "goals": ["Explore"]}}
    second = {"name": "Ada", "type": kind, "description": "Dead.", "first_appearance_chapter": 1, "scene_index": 1,
              "attributes": {"id": "exact-17", "status": "Dead", "category": kind, "traits": ["remembered"], "key_elements": ["New fact"]}}
    original = deepcopy([first, second])
    merged = consolidate_scene_extractions([{collection: [first]}, {collection: [second]}])[collection][0]
    assert merged["attributes"]["status"] == "Dead"
    assert merged["description"] == "Dead."
    assert merged["scene_index"] == 1
    assert merged["attributes"]["key_elements"] == ["New fact"]
    assert merged["attributes"]["goals"] == ["Explore"]
    assert merged["attributes"]["scene_assertions"] == original
    assert [first, second] == original
    entity = ExtractedEntity.model_validate(merged)
    if kind == "Character":
        profile = _convert_to_character_profiles([entity], {}, 1)[0]
        assert profile.status == "Dead"
        assert profile.updates["scene_assertions"] == original
    else:
        item = _convert_to_world_items([entity], {}, 1)[0]
        assert json.loads(item.additional_properties["scene_assertions"]) == original
        _, parameters = NativeCypherBuilder.world_item_upsert_cypher(item, 1)
        assert json.loads(parameters["additional_props"]["scene_assertions"]) == original


async def test_chronology_survives_real_scene_entrypoint_and_content_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = selected_scene(tmp_path)
    manager = ContentManager(str(tmp_path))
    state["scene_drafts_ref"] = manager.save_json([SCENE, SCENE, SCENE], "scene_drafts", "chapter_1", 2)
    replies = responses() + responses() + responses()
    replies[0]["character_updates"]["Father O'Brien"]["description"] = "Long old account of an alive character standing in the hall."
    replies[0]["character_updates"]["Father O'Brien"]["status"] = "Alive"
    replies[4]["character_updates"]["Father O'Brien"]["description"] = "Dead."
    replies[4]["character_updates"]["Father O'Brien"]["status"] = "Dead"
    replies[8]["character_updates"]["Father O'Brien"]["description"] = "Remembered."
    replies[8]["character_updates"]["Father O'Brien"]["status"] = "Dead"
    result, _ = await run_extraction(monkeypatch, state, replies)
    assert result["extraction_status"] == "complete"
    row = manager.load_json_strict(result["extracted_entities_ref"])["characters"][0]
    assert row["attributes"]["status"] == "Dead"
    assert row["description"] == "Remembered."
    assert [item["attributes"]["status"] for item in row["attributes"]["scene_assertions"]] == ["Alive", "Dead", "Dead"]
    assert [item["scene_index"] for item in row["attributes"]["scene_assertions"]] == [0, 1, 2]


def test_repeated_relationship_keeps_latest_description_and_all_source_assertions() -> None:
    first = {"source_name": "Ada", "target_name": "Bob", "relationship_type": "TRUSTS", "description": "A long early account.", "chapter": 1, "scene_index": 0}
    second = {**first, "description": "Qualified trust.", "scene_index": 1}
    merged = consolidate_scene_extractions([{"relationships": [first]}, {"relationships": [second]}])["relationships"][0]
    assert merged["description"] == "Qualified trust."
    assert merged["scene_index"] == 1
    assert merged["scene_assertions"] == [first, second]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("kind", ["Character", "Location"])
def test_bulk_models_preserve_same_target_multirelationships(kind: str, reverse: bool) -> None:
    rows = [{"target_name": "Bob", "type": "TRUSTS", "description": "Trust"}, {"target_name": "Bob", "type": "ALLIES_WITH", "description": "Alliance"},
            {"target_name": "Cara", "type": "KNOWS", "description": "Acquaintance"}]
    if reverse:
        rows.reverse()
    node = {"name": "Ada", "id": "exact-17", "category": "location"}
    profile = CharacterProfile.from_dict_record({"c": node, "relationships": rows}) if kind == "Character" else WorldItem.from_dict_record({"w": node, "relationships": rows})
    assert profile.relationships == {
        "Bob": [{"type": "ALLIES_WITH", "description": "Alliance"}, {"type": "TRUSTS", "description": "Trust"}],
        "Cara": {"type": "KNOWS", "description": "Acquaintance"},
    }


@pytest.mark.usefixtures("owned_graph_cache")
async def test_bulk_context_and_individual_character_reads_have_equivalent_projection(offline_graph_reads: FakeNeo4jManager) -> None:
    flat = [{"target_name": "Bob", "type": "TRUSTS", "description": "Trust"}, {"target_name": "Bob", "type": "ALLIES_WITH", "description": "Alliance"}]
    node = {"id": "exact-17", "name": "Ada", "personality_description": "Explorer", "traits": ["brave"], "status": "Alive"}
    nested = [{"target_name": row["target_name"], "rel_type": row["type"], "rel_props": {"description": row["description"], "source_profile_managed": True}} for row in flat]
    offline_graph_reads.configure_response(r"MATCH \(c:Character \{", [{"c": node, "traits": ["brave"], "relationships": nested}])
    offline_graph_reads.configure_response(r"MATCH \(c:Character\)", [{"c": node, "relationships": list(reversed(flat))}])
    by_name = await character_queries.get_character_profile_by_name("Ada")
    by_id = await character_queries.get_character_profile_by_id("exact-17")
    bulk = await character_queries.get_character_profiles()
    context = await character_queries.get_characters_for_chapter_context_native(2)
    assert by_name is not None and by_id is not None
    assert len(bulk) == len(context) == 1
    assert bulk[0].model_dump() == context[0].model_dump() == by_name.model_dump() == by_id.model_dump()
    assert len(bulk[0].relationships["Bob"]) == 2
