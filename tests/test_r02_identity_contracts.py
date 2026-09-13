"""Identity conservation across producer, consolidation, conversion and query inputs."""
from typing import Any

import pytest

from core.langgraph.nodes.commit_entity_conversion import _convert_to_character_profiles, _convert_to_world_items
from core.langgraph.nodes.commit_node import _build_relationship_statements
from core.langgraph.nodes.commit_validation import _deduplicate_entity_list
from core.langgraph.nodes.scene_extraction_normalization import consolidate_scene_extractions
from core.langgraph.nodes.scene_extraction_parsing import normalize_triple_entities
from prompts.prompt_renderer import render_prompt
from tests.test_r02_extraction_identity import entity, relationship


@pytest.mark.parametrize("template", ["extract_relationships", "extract_characters", "extract_locations", "extract_events"])
def test_producers_share_exact_identity_and_empty_result_guidance(template: str) -> None:
    prompt = render_prompt(f"knowledge_agent/{template}.j2", {
        "chapter_number": 1, "novel_title": "Synthetic", "novel_genre": "Fantasy", "protagonist": "Mara",
        "chapter_text": "Mara enters King's Cross.", "canonical_relationship_types": [],
    })
    assert "exact spelling, capitalization and punctuation" in prompt
    assert "Do not invent" in prompt
    assert "The Hague" in prompt
    assert "King's Cross" in prompt
    assert "empty" in prompt


@pytest.mark.parametrize("first_id,second_id", [("place-17", "place-18"), ("place-17", "")])
def test_consolidation_rejects_explicit_id_conflicts(first_id: str, second_id: str) -> None:
    first = entity("The Hague", "Location", id=first_id).model_dump()
    second = entity("The Hague", "Location", id=second_id).model_dump()
    with pytest.raises(ValueError, match="identity|ID"):
        consolidate_scene_extractions([{"world_items": [first]}, {"world_items": [second]}])


def test_consolidation_carries_explicit_id_from_shorter_record_without_mutation() -> None:
    first = entity("The Hague", "Location", id="place-17").model_dump()
    second = entity("The Hague", "Location").model_dump()
    second["description"] = "A much longer description of the same named place"
    result = consolidate_scene_extractions([{"world_items": [first]}, {"world_items": [second]}])
    assert result["world_items"][0]["attributes"]["id"] == "place-17"
    assert second["attributes"] == {}


def test_commit_does_not_discard_conflicting_same_name_entity() -> None:
    with pytest.raises(ValueError, match="Conflicting"):
        _deduplicate_entity_list([entity("The Lantern", "Location"), entity("The Lantern", "Item")])


def test_conversion_rejects_category_label_mismatch_without_explicit_id() -> None:
    with pytest.raises(ValueError, match="category|label"):
        _convert_to_world_items([entity("The Hague", "Location", category="Weapon")], {"The Hague": "place-17"}, 1)


async def test_relationship_rejects_ambiguous_same_name_types() -> None:
    with pytest.raises(ValueError, match="identity|ambiguous|Ambiguous"):
        await _build_relationship_statements(
            [relationship("Mara", "The Lantern")], [entity("Mara", "Character")],
            [entity("The Lantern", "Location"), entity("The Lantern", "Item")], {}, {}, 1, False,
        )


async def test_relationship_rejects_profile_target_id_conflict() -> None:
    characters = [entity("Mara", "Character", relationships={"Silas": {"type": "MENTORS", "target_id": "wrong-id"}}),
                  entity("Silas", "Character", id="person-17")]
    with pytest.raises(ValueError, match="identity|ID"):
        await _build_relationship_statements([], characters, [], {}, {}, 1, False)


@pytest.mark.parametrize("bad", [{"subject": {"name": "Mara"}}, {"subject": "Mara"}, {"subject": 17}])
def test_triple_conversion_does_not_forge_missing_or_nested_fields(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        normalize_triple_entities(bad)


@pytest.mark.parametrize("identifier", ["", "  ", None, 17])
@pytest.mark.parametrize("kind", ["Character", "Location"])
def test_conversion_never_repairs_invalid_explicit_id(identifier: Any, kind: str) -> None:
    converter = _convert_to_character_profiles if kind == "Character" else _convert_to_world_items
    with pytest.raises(ValueError):
        converter([entity("The Hague", kind, id=identifier)], {}, 1)
