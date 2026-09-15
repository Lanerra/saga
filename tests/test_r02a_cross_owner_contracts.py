"""Prepared RED contracts for coordinator-owned state and native writer seams.

These intentionally remain failing until the cross-owner dependencies in
R02A/findings.json are resolved; they are not evidence of repaired behavior.
"""

from pathlib import Path

import pytest

from core.langgraph.content_manager import ContentManager, get_extracted_relationships
from core.langgraph.state import ExtractedRelationship
from data_access.cypher_builders.native_builders import NativeCypherBuilder
from models.kg_models import CharacterProfile, WorldItem


def test_relationship_content_model_retains_explicit_identity_and_scene_provenance(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))
    row = {"source_name": "Ada", "target_name": "Named Sword", "source_type": "Character", "target_type": "Item", "source_id": "person-17", "target_id": "item-17",
           "relationship_type": "WIELDS", "description": "Synthetic assertion", "chapter": 1, "scene_index": 0,
           "scene_assertions": [{"scene_index": 0, "description": "Synthetic assertion"}]}
    reference = manager.save_json([row], "extracted_relationships", "chapter_1", 1)
    loaded = get_extracted_relationships({"project_dir": str(tmp_path), "extracted_relationships_ref": reference}, manager)
    # commit_to_graph constructs this model from the content loader's dictionaries.
    payload = ExtractedRelationship(**loaded[0]).model_dump()
    assert {key: payload.get(key) for key in ("source_id", "target_id", "scene_index", "scene_assertions")} == {
        key: row[key] for key in ("source_id", "target_id", "scene_index", "scene_assertions")
    }


@pytest.mark.parametrize("kind", ["Character", "Location"])
def test_native_writer_preserves_declared_multi_relationship_projection(kind: str) -> None:
    predicates = ["ALLIES_WITH", "TRUSTS"] if kind == "Character" else ["CONTAINS_LOCATION", "BORDERS"]
    relationships = {"Bob": [{"type": predicate, "description": "Synthetic evidence"} for predicate in predicates]}
    if kind == "Character":
        profile = CharacterProfile(name="Ada", id="person-17", relationships=relationships)
        _, parameters = NativeCypherBuilder.character_upsert_cypher(profile, 1)
    else:
        item = WorldItem(name="Named Place", id="place-17", category="location", relationships=relationships)
        _, parameters = NativeCypherBuilder.world_item_upsert_cypher(item, 1)
    assert {row["rel_type"] for row in parameters["relationship_data"]} == set(predicates)
