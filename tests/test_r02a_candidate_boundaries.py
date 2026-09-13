"""Closed-set edge cases using real catalogs, content files and transport fakes."""

from pathlib import Path
from typing import Any

import pytest

import config
from core.langgraph.content_manager import ContentManager
from core.langgraph.initialization.catalog import select_catalog
from core.langgraph.nodes import scene_extraction
from core.langgraph.nodes.scene_extraction_validation import _validate_entity_with_spacy, scene_identity_candidates, validate_named_entity
from core.langgraph.state import NarrativeState
from models.kg_models import WorldItem
from tests.test_r02a_named_candidates import SCENE, responses, run_extraction, selected_scene
from tests.test_r08t_migration_contracts import selected_authority_state


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_nlp_never_enlarges_or_shrinks_eligible_set(tmp_path: Path, enabled: bool) -> None:
    candidates = scene_identity_candidates(select_catalog(selected_scene(tmp_path)), SCENE)
    assert set(candidates) == {"Father O'Brien", "The Hague", "King's Cross", "The Blackwood Family"}
    for name in candidates:
        assert validate_named_entity(name, candidates) == name
        with config.bind_settings(config.snapshot_settings().model_copy(update={"ENABLE_ENTITY_VALIDATION": enabled})):
            assert _validate_entity_with_spacy(SCENE, name, candidates)
            assert not _validate_entity_with_spacy("Nothing here.", name, candidates)
    assert not _validate_entity_with_spacy(SCENE, "Neighbors", candidates)
    assert not _validate_entity_with_spacy(SCENE, "Hague", candidates)
    assert not _validate_entity_with_spacy(SCENE, "Kings Cross", candidates)


async def test_same_word_can_be_explicitly_cataloged_as_a_named_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = selected_named_scene(tmp_path, "Neighbors")
    manager = ContentManager(str(tmp_path))
    catalog = select_catalog(state)
    group = next(candidate for candidate in catalog.candidates("Character") if candidate["name"] == "Neighbors")
    replies = responses()
    replies[0]["character_updates"]["Neighbors"] = {"description": "The selected named organization", "traits": [], "status": "Present", "relationships": {}}
    replies[3]["kg_triples"].append({"subject": "Neighbors", "object_entity": "Father O'Brien", "predicate": "ALLIES_WITH", "description": "Synthetic explicit support."})
    result, requests = await run_extraction(monkeypatch, state, replies)
    assert result["extraction_status"] == "complete"
    assert len(requests) == 4
    rows = manager.load_json_strict(result["extracted_relationships_ref"])
    assert rows[1]["source_id"] == group["id"]
    assert rows[1]["source_name"] == "Neighbors"


@pytest.mark.parametrize("producer,label,name", [(0, "character_updates", "The Hague"), (1, "Location", "Father O'Brien"), (2, "Event", "The Hague")])
async def test_selected_name_cannot_be_retyped_by_entity_producer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, producer: int, label: str, name: str) -> None:
    state = selected_scene(tmp_path)
    replies = responses()
    mapping = replies[producer][label] if producer == 0 else replies[producer]["world_updates"][label]
    mapping[name] = {"description": "Wrong entity kind"}
    result, _ = await run_extraction(monkeypatch, state, replies)
    assert result["extraction_status"] == "failed"
    assert result["extracted_entities_ref"] is None
    assert "ineligible" in result["last_error"]


@pytest.mark.parametrize("name", ["Absent Name", "Hague", "Kings Cross", "Father O’Brian", "NEIGHBORS"])
async def test_outside_scene_or_nonliteral_name_fails_despite_producer_assertion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    state = selected_scene(tmp_path)
    replies = responses()
    replies[3]["kg_triples"][0]["object_entity"] = name
    result, _ = await run_extraction(monkeypatch, state, replies)
    assert result["extraction_status"] == "failed"
    assert result["extracted_relationships_ref"] is None


@pytest.mark.parametrize("invalid_row", [False, True])
async def test_empty_candidate_set_is_explicit_not_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_row: bool) -> None:
    state = selected_scene(tmp_path, "Neighbors gathered at the community hall.")
    replies: list[dict[str, Any]] = [{"character_updates": {}}, {"world_updates": {"Location": {}}}, {"world_updates": {"Event": {}}}, {"kg_triples": []}]
    if invalid_row:
        replies[3] = {"kg_triples": [{"subject": "Neighbors", "object_entity": "Neighbors", "predicate": "ALLIES_WITH", "description": "Invalid"}]}
    result, requests = await run_extraction(monkeypatch, state, replies)
    assert len(requests) == 4
    schema = requests[3]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["kg_triples"]["maxItems"] == 0
    assert result["extraction_status"] == ("failed" if invalid_row else "complete")
    if not invalid_row:
        assert ContentManager(str(tmp_path)).load_json_strict(result["extracted_relationships_ref"]) == []


def selected_named_scene(tmp_path: Path, additional_name: str) -> NarrativeState:
    state = selected_authority_state(
        tmp_path, ("Father O'Brien", "The Blackwood Family", "Absent Name", additional_name),
        world_items=tuple(WorldItem(name=name, category="location", description="Synthetic named place", id=identity)
                          for name, identity in [("The Hague", "place-exact-17"), ("King's Cross", "place-exact-18")]),
    )
    state["current_chapter"] = 1
    state["scene_drafts_ref"] = ContentManager(str(tmp_path)).save_json([SCENE], "scene_drafts", "chapter_1", 1)
    return state


async def test_ambiguous_catalog_name_fails_before_scene_producers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = selected_named_scene(tmp_path, "The Hague")
    catalog = select_catalog(state)
    candidates = [candidate for candidate in catalog.candidates("Character", "Location") if candidate["name"] == "The Hague"]
    assert {candidate["label"] for candidate in candidates} == {"Character", "Location"}
    assert len({candidate["id"] for candidate in candidates}) == 2
    result, requests = await run_extraction(monkeypatch, state, responses())
    assert result["extraction_status"] == "failed"
    assert "Ambiguous eligible" in result["last_error"]
    assert requests == []


async def test_missing_catalog_direct_scene_entrypoint_fails_all_slots() -> None:
    result = await scene_extraction.extract_from_scene(SCENE, 0, 1, "Synthetic", "Fantasy", "Father O'Brien", "synthetic")
    assert set(result) == {"extraction_status", "extraction_outcomes"}
    assert result["extraction_status"] == "failed"
    assert len(result["extraction_outcomes"]) == 4
    assert all(row["status"] == "failed" and "eligible identity catalog" in row["error"] for row in result["extraction_outcomes"])


async def test_failed_later_scene_does_not_publish_earlier_valid_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = selected_scene(tmp_path)
    manager = ContentManager(str(tmp_path))
    state["scene_drafts_ref"] = manager.save_json([SCENE, SCENE], "scene_drafts", "chapter_1", 2)
    invalid = responses()
    invalid[3]["kg_triples"][0]["subject"] = "Neighbors"
    result, requests = await run_extraction(monkeypatch, state, responses() + invalid)
    assert len(requests) == 8
    assert result["extraction_status"] == "failed"
    assert result["extracted_entities_ref"] is None
    assert result["extracted_relationships_ref"] is None
    assert len(result["extraction_outcomes"]) == 8
    assert all(row["status"] == "succeeded" for row in result["extraction_outcomes"][:4])
    assert result["extraction_outcomes"][-1]["status"] == "failed"
