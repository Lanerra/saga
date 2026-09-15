"""Positive controls paired with fail-closed integration migrations."""

import json
from typing import Any

import pytest

from core.langgraph.initialization.chapter_outline_node import _parse_chapter_outline
from core.langgraph.nodes.commit_validation import _deduplicate_entity_list
from core.langgraph.nodes.scene_extraction_normalization import consolidate_scene_extractions
from core.langgraph.nodes.scene_planning_node import _parse_scene_plan_json_from_llm_response
from core.langgraph.state import ExtractedEntity


@pytest.mark.parametrize("count", [3, 4, 5])
def test_chapter_beats_preserved_without_truncation(count: int) -> None:
    payload = {"scene_description": "Ada's arrival—unchanged.", "key_beats": [f"Beat {number}" for number in range(count)], "plot_point": "Ada chooses."}
    response = json.dumps(payload)
    result = _parse_chapter_outline(response, 2, 1)
    assert result == {**payload, "chapter_number": 2, "act_number": 1, "raw_text": response, "generated_at": "on_demand"}


@pytest.mark.parametrize("count", [0, 1, 2, 6, 20])
def test_invalid_chapter_beat_count_is_not_repaired(count: int) -> None:
    payload = {"scene_description": "Ada arrives.", "key_beats": [f"Beat {number}" for number in range(count)], "plot_point": "Ada chooses."}
    with pytest.raises(ValueError):
        _parse_chapter_outline(json.dumps(payload), 2, 1)
    assert len(payload["key_beats"]) == count


def test_identical_commit_duplicates_preserve_first_record_and_order() -> None:
    alice = ExtractedEntity(name="Alice", type="Character", description="Scout", first_appearance_chapter=1, attributes={"id": "literal-A"})
    bob = ExtractedEntity(name="Bob", type="Character", description="Guide", first_appearance_chapter=1, attributes={"id": "literal-B"})
    duplicate = alice.model_copy(deep=True)
    assert _deduplicate_entity_list([alice, bob, duplicate]) == [alice, bob]
    assert _deduplicate_entity_list([]) == []


@pytest.mark.parametrize("collection", ["characters", "world_items"])
def test_consolidation_identical_duplicates_and_empty(collection: str) -> None:
    entity: dict[str, Any] = {"name": "Crossing", "type": "Character" if collection == "characters" else "Location", "description": "Named entity", "attributes": {"id": "literal-ID"}}
    relationship = {"source_name": "Ada", "target_name": "Crossing", "relationship_type": "LOCATED_AT"}
    result = consolidate_scene_extractions([{collection: [entity, entity.copy()], "relationships": [relationship, relationship.copy()]}])
    assert result[collection] == [{**entity, "attributes": {**entity["attributes"], "scene_assertions": [entity, entity]}}]
    assert result["relationships"] == [{**relationship, "scene_assertions": [relationship, relationship]}]
    assert consolidate_scene_extractions([]) == {"characters": [], "world_items": [], "relationships": []}


@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_same_name_ids_reject_whole_consolidation(reverse: bool) -> None:
    first: dict[str, Any] = {"name": "Crossing", "type": "Location", "description": "A ford", "attributes": {"id": "first"}}
    second = {**first, "description": "A longer description", "attributes": {"id": "second"}}
    items = [second, first] if reverse else [first, second]
    with pytest.raises(ValueError, match="Conflicting entity identity"):
        consolidate_scene_extractions([{"world_items": items}])
    assert [item["attributes"]["id"] for item in items] == (["second", "first"] if reverse else ["first", "second"])


@pytest.mark.parametrize("invalid_beats", ["Ada opens the door", [1], [""], ["   "]])
def test_scene_beats_strict_rejection_with_valid_control(invalid_beats: Any) -> None:
    scene = {"title": "Arrival", "pov_character": "Ada", "setting": "Harbor", "characters": ["Ada"], "plot_point": "Arrival", "conflict": "Locked door", "outcome": "Door opens", "beats": ["Ada opens the door"]}
    assert _parse_scene_plan_json_from_llm_response(json.dumps([scene])) == [scene]
    with pytest.raises(ValueError, match="Scene plan contract violation"):
        _parse_scene_plan_json_from_llm_response(json.dumps([{**scene, "beats": invalid_beats}]))
