"""Production statement contracts; graph semantics are exercised by the B17 engine matrix."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import config
from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.commit_node import _build_relationship_statements, commit_to_graph
from core.langgraph.state import ExtractedEntity, ExtractedRelationship, NarrativeState
from core.service_context import get_services
from data_access.cypher_builders.native_builders import NativeCypherBuilder, relationship_statement
from models.kg_models import CharacterProfile, WorldItem


@pytest.mark.parametrize("label", ["Character", "Location", "Item", "Event"])
async def test_commit_preserves_supplied_canonical_ids(label: str) -> None:
    entities = [ExtractedEntity(name=name, type=label, description="Synthetic entity", first_appearance_chapter=1, attributes={"id": identifier})
                for name, identifier in [("Renamed source", "canonical-source"), ("Renamed target", "canonical-target")]]
    relationships = [ExtractedRelationship(source_name="Renamed source", target_name="Renamed target", source_type=label, target_type=label,
                                          relationship_type="RELATED_TO", description="Synthetic relation", chapter=2)]
    statements = await _build_relationship_statements(relationships, entities if label == "Character" else [], entities if label != "Character" else [], {}, {}, 2, False)
    assert len(statements) == 2
    parameters = statements[1][1]
    assert (parameters["subject_id"], parameters["object_id"], parameters["subject_label"], parameters["object_label"]) == ("canonical-source", "canonical-target", label, label)


@pytest.mark.parametrize("label", ["Character", "Location", "Item", "Event"])
async def test_profile_channel_is_independent_of_entity_upsert(label: str) -> None:
    entity = ExtractedEntity(name="Source", type=label, description="Synthetic", first_appearance_chapter=1,
                             attributes={"id": "canonical-source", "relationships": {"Target": {"type": "RELATED_TO", "description": "Profile", "target_id": "canonical-target", "target_label": label}}})
    statements = await _build_relationship_statements([], [entity] if label == "Character" else [], [entity] if label != "Character" else [], {}, {}, 2, False)
    assert len(statements) == 2
    parameters = statements[1][1]
    assert (parameters["subject_id"], parameters["object_id"], parameters["assertion_origin"]) == ("canonical-source", "canonical-target", "chapter_profile")


@pytest.mark.parametrize("identifier", ["", " ", 7, []])
def test_invalid_supplied_identity_fails_before_query(identifier: Any) -> None:
    with pytest.raises(ValueError, match="Invalid canonical entity ID"):
        relationship_statement({"name": "Source", "type": "Location", "id": identifier}, "RELATED_TO", {"name": "Target", "type": "Location"}, 2, origin="import", provisional=False)


def test_native_writer_preserves_source_and_target_identity() -> None:
    character = CharacterProfile(name="Renamed source", id="canonical-source", relationships={"Renamed target": {"target_id": "canonical-target", "type": "KNOWS", "description": "Profile"}})
    _, parameters = NativeCypherBuilder.character_upsert_cypher(character, 2)
    assert parameters["id"] == "canonical-source"
    assert parameters["assertion_origin"] == "profile"
    assert parameters["relationship_data"] == [{"target_name": "Renamed target", "target_id": "canonical-target", "target_label": "Character", "rel_type": "KNOWS", "description": "Profile", "chapter_added": 2, "assertion_origin": "profile", "properties": {"type": "KNOWS", "description": "Profile", "source_profile_managed": True}}]
    world = WorldItem(id="canonical-world", name="Renamed world", category="Location")
    _, parameters = NativeCypherBuilder.world_item_upsert_cypher(world, 2)
    assert (parameters["id"], parameters["primary_label"], parameters["assertion_origin"]) == ("canonical-world", "Location", "profile")


@pytest.mark.parametrize("field", ["id", "name"])
def test_additional_properties_cannot_overwrite_identity(field: str) -> None:
    item = WorldItem(id="canonical-world", name="World", category="Location", additional_properties={field: "replacement"})
    with pytest.raises(ValueError, match="^Additional properties cannot override canonical identity$"):
        NativeCypherBuilder.world_item_upsert_cypher(item, 2)


@pytest.mark.parametrize("label", ["Character", "Location", "Item", "Event"])
@pytest.mark.parametrize("identifier", [None, "", " ", 7, False, []])
async def test_public_admission_rejects_malformed_explicit_identity(
    label: str, identifier: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(vars(config), "ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    manager = ContentManager(str(tmp_path))
    entity = {"name": "Protected", "type": label, "description": "Candidate",
              "first_appearance_chapter": 2, "attributes": {"id": identifier}}
    state: NarrativeState = {
        "project_dir": str(tmp_path), "current_chapter": 2,
        "draft_ref": manager.save_text("Synthetic draft.", "draft", "2"),
        "extracted_entities_ref": manager.save_json(
            {"characters": [entity] if label == "Character" else [],
             "world_items": [] if label == "Character" else [entity]}, "entities", "2"),
    }
    reads: list[str] = []
    batches: list[Any] = []

    async def existing_names(query: str, parameters: Any = None) -> list[dict[str, Any]]:
        reads.append(query)
        return [{"name": "protected"}]

    async def record_batch(statements: Any) -> None:
        batches.append(statements)

    monkeypatch.setattr(get_services().database, 'execute_read_query', existing_names)
    monkeypatch.setattr(get_services().database, 'execute_cypher_batch', record_batch)
    result = await commit_to_graph(state)
    assert result == {"current_node": "commit_to_graph", "last_error": "Commit to graph failed: Invalid canonical entity ID",
                      "has_fatal_error": True, "error_node": "commit"}
    assert reads == []
    assert batches == []


@pytest.mark.parametrize("label", ["Character", "Location", "Item", "Event"])
@pytest.mark.run_settings(ENABLE_ENTITY_EMBEDDING_PERSISTENCE=True, EXPECTED_EMBEDDING_DIM=2)
async def test_public_alias_keeps_profile_upsert_out_of_embedding_batch(
    label: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ContentManager(str(tmp_path))
    entity = {"name": "Alias", "type": label, "description": "Scene description",
              "first_appearance_chapter": 2, "attributes": {"id": "canonical", "category": label,
                  "relationships": {"Friend": {"type": "RELATED_TO", "description": "Chapter assertion",
                                                 "target_id": "friend", "target_label": label}}}}
    state: NarrativeState = {
        "project_dir": str(tmp_path), "current_chapter": 2,
        "draft_ref": manager.save_text("Synthetic draft.", "draft", "2"),
        "extracted_entities_ref": manager.save_json(
            {"characters": [entity] if label == "Character" else [],
             "world_items": [] if label == "Character" else [entity]}, "entities", "2"),
    }
    batches: list[Any] = []
    inputs: list[list[str]] = []

    async def read_identity(query: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        if "admission_entities" in parameters:
            assert parameters == {"admission_entities": [{"index": 0, "label": label, "id": "canonical", "name": "Alias"}]}
            return [{"index": 0, "existing": True}]
        if "entities" in parameters:
            assert parameters["entities"] == [{"index": 0, "label": label, "id": "canonical", "name": "Alias",
                                               "category": "" if label == "Character" else label, "description": "Scene description"}]
            return [{"key": 0, "id": "canonical", "existing_hash": None}]
        assert parameters == {}
        return [{"name": "original"}, {"name": "friend"}]

    async def provider(texts: list[str]) -> list[Any]:
        inputs.append(texts)
        return [np.array([0.25, 0.75])]

    async def record_batch(statements: Any) -> None:
        batches.append(statements)

    monkeypatch.setattr(get_services().database, 'execute_read_query', read_identity)
    monkeypatch.setattr(get_services().database, 'execute_cypher_batch', record_batch)
    monkeypatch.setattr(get_services().language_model, 'async_get_embeddings_batch', provider)
    assert await commit_to_graph(state) == {"current_node": "commit_to_graph", "last_error": None, "has_fatal_error": False}
    assert inputs == [["Alias\nScene description" if label == "Character" else f"Alias\n{label}\nScene description"]]
    assert len(batches) == 1
    statements = batches[0]
    assert len(statements) == 5
    assert statements[0][1] == {"admission_entities": [{"index": 0, "label": label, "id": "canonical", "name": "Alias", "existing": True}]}
    assert statements[1][1] == {"chapter": 2}
    assert statements[2][1]["identity"] == {"label": label, "id": "canonical", "name": "Alias"}
    assert statements[2][1]["vector"] == [0.25, 0.75]
    assert (statements[3][1]["subject_id"], statements[3][1]["object_id"], statements[3][1]["assertion_origin"]) == ("canonical", "friend", "chapter_profile")
    assert statements[4][1]["chapter_number_param"] == 2
    assert [parameters for _, parameters in statements if "trait_data" in parameters or "additional_props" in parameters] == []
