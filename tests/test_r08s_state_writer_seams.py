"""Synthetic content/model/native boundaries; no live graph or model evidence."""

import json
from pathlib import Path
from typing import Any

import pytest

import config
from core.langgraph.content_manager import ContentManager, get_extracted_relationships
from core.langgraph.nodes.commit_entity_conversion import _convert_to_character_profiles, _convert_to_world_items
from core.langgraph.nodes.commit_node import _build_entity_persistence_statements, _build_relationship_statements
from core.langgraph.state import ExtractedEntity, ExtractedRelationship
from data_access.cypher_builders.native_builders import NativeCypherBuilder
from models.kg_models import CharacterProfile, WorldItem, project_relationships_by_target


def relationship_row() -> dict[str, Any]:
    return {"source_name": "Ada", "target_name": "Named Sword", "source_type": "Character", "target_type": "Item",
            "source_id": "Person:Exact-17", "target_id": "Item:Exact-17", "relationship_type": "WIELDS",
            "description": "Later assertion", "chapter": 2, "scene_index": 1,
            "scene_assertions": [{"scene_index": 0, "description": "Earlier assertion"}, {"scene_index": 1, "description": "Later assertion"}]}


async def test_content_to_real_commit_builder_retains_identity_and_provenance(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))
    row = relationship_row()
    reference = manager.save_json([row], "extracted_relationships", "chapter_2", 1)
    loaded = get_extracted_relationships({"project_dir": str(tmp_path), "extracted_relationships_ref": reference}, manager)
    model = ExtractedRelationship(**loaded[0])
    statements = await _build_relationship_statements([model], [], [], {}, {}, 2, False)
    query, parameters = statements[1]
    assert parameters["subject_id"] == row["source_id"]
    assert parameters["object_id"] == row["target_id"]
    assert parameters["subject_name"] == "Ada"
    assert parameters["object_name"] == "Named Sword"
    assert parameters["chapter"] == 2
    assert parameters["assertion_origin"] == "chapter_extraction"
    properties = parameters["relationship_properties"]
    assert properties["scene_index"] == 1
    assert json.loads(properties["scene_assertions"]) == row["scene_assertions"]
    assert "$relationship_properties" in query


@pytest.mark.parametrize("field", ["source_id", "target_id"])
@pytest.mark.parametrize("value", ["", " ", " Person:Exact-17", 17, False])
def test_explicit_identity_is_never_normalized_or_discarded(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        ExtractedRelationship(**{**relationship_row(), field: value})


@pytest.mark.parametrize("changes", [
    {"scene_index": -1}, {"scene_index": True}, {"scene_index": "1"},
    {"scene_assertions": [{"scene_index": 0, "description": "Evidence", "source_id": "conflicting"}]},
    {"scene_assertions": [{"scene_index": 2, "description": "Future assertion"}]},
    {"unrecognized_identity": "no silent extras"},
])
def test_invalid_provenance_is_not_silently_ignored(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ExtractedRelationship(**{**relationship_row(), **changes})


async def test_legacy_optional_fields_remain_absent_from_native_properties() -> None:
    row = {key: value for key, value in relationship_row().items() if key not in {"source_id", "target_id", "scene_index", "scene_assertions"}}
    model = ExtractedRelationship(**row)
    assert model.model_dump(exclude_unset=True) == row
    _, parameters = (await _build_relationship_statements([model], [], [], {}, {}, 2, False))[1]
    assert parameters["subject_id"] is None
    assert parameters["object_id"] is None
    assert "scene_index" not in parameters["relationship_properties"]
    assert "scene_assertions" not in parameters["relationship_properties"]


async def test_explicit_relationship_identity_conflict_is_rejected() -> None:
    entity = ExtractedEntity(name="Ada", type="Character", description="Protected", first_appearance_chapter=0, attributes={"id": "other-id"})
    with pytest.raises(ValueError, match="ID conflicts"):
        await _build_relationship_statements([ExtractedRelationship(**relationship_row())], [entity], [], {}, {}, 2, False)


async def test_explicit_identity_does_not_authorize_name_alias_mapping() -> None:
    with pytest.raises(ValueError, match="name"):
        await _build_relationship_statements([ExtractedRelationship(**relationship_row())], [], [], {"Ada": "Other Ada"}, {}, 2, False)


@pytest.mark.parametrize("kind", ["Character", "Location"])
def test_reader_projection_expands_every_predicate_and_chapter(kind: str) -> None:
    predicates = ["ALLIES_WITH", "TRUSTS"] if kind == "Character" else ["BORDERS", "CONTAINS_LOCATION"]
    records = [{"type": predicate, "description": f"Evidence {chapter}", "target_id": "Target:Exact-2", "target_label": kind,
                "chapter_added": chapter, "assertion_origin": "profile", "scene_index": chapter,
                "scene_assertions": json.dumps([{"scene_index": chapter, "description": f"Evidence {chapter}"}])}
               for predicate, chapter in [(predicates[0], 0), (predicates[0], 1), (predicates[1], 1)]]
    projection = project_relationships_by_target({"Bob": records})
    if kind == "Character":
        query, parameters = NativeCypherBuilder.character_upsert_cypher(CharacterProfile(name="Ada", id="ada", relationships=projection), 5)
    else:
        query, parameters = NativeCypherBuilder.world_item_upsert_cypher(WorldItem(name="Place", id="place", category="location", relationships=projection), 5)
    rows = parameters["relationship_data"]
    assert len(rows) == 3
    assert {(row["rel_type"], row["chapter_added"]) for row in rows} == {(record["type"], record["chapter_added"]) for record in records}
    assert all(row["target_id"] == "Target:Exact-2" for row in rows)
    assert all(row["assertion_origin"] == "profile" for row in rows)
    assert all(row["properties"]["scene_assertions"] for row in rows)
    assert "rel_data.chapter_added" in query
    assert "rel_data.assertion_origin" in query


@pytest.mark.parametrize("kind", ["Character", "Location"])
@pytest.mark.parametrize("information", ["TRUSTS", [], [{"description": "missing predicate"}], [{"type": "trusts"}], [{"type": "TRUSTS", "target_id": " bad"}]])
def test_native_writer_rejects_undeclared_relationship_shapes(kind: str, information: Any) -> None:
    with pytest.raises(ValueError):
        if kind == "Character":
            NativeCypherBuilder.character_upsert_cypher(CharacterProfile(name="Ada", relationships={"Bob": information}), 1)
        else:
            NativeCypherBuilder.world_item_upsert_cypher(WorldItem(name="Place", id="place", category="location", relationships={"Bob": information}), 1)


async def test_entity_chronology_survives_conversion_without_overwriting_protected_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(vars(config), "ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    history = [{"name": "Ada", "scene_index": 0, "description": "First"}, {"name": "Ada", "scene_index": 1, "description": "Later"}]
    character = ExtractedEntity(name="Ada", type="Character", description="Later", first_appearance_chapter=0, attributes={"id": "ada", "scene_assertions": history})
    world = ExtractedEntity(name="Place", type="Location", description="Later", first_appearance_chapter=0,
                            attributes={"id": "place", "scene_assertions": history, "category": "location"})
    characters = _convert_to_character_profiles([character], {}, 2)
    items = _convert_to_world_items([world], {}, 2)
    statements = await _build_entity_persistence_statements(characters, items, 2)
    assert statements[0][1]["created_chapter"] == 0
    assert statements[1][1]["created_chapter"] == 0
    assert json.loads(statements[0][1]["domain_properties"]["scene_assertions"]) == history
    assert json.loads(statements[1][1]["additional_props"]["scene_assertions"]) == history
    assert await _build_entity_persistence_statements(characters, items, 2, protected_identities=frozenset({("Character", "ada"), ("Location", "place")})) == []
    assert characters[0].updates["scene_assertions"] == history
