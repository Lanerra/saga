"""Offline preparation and coordinator-only fresh Neo4j readback; no model requests."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from core.langgraph.content_manager import ContentManager, get_extracted_relationships
from core.langgraph.nodes.commit_node import _build_relationship_statements
from core.langgraph.state import ExtractedRelationship
from data_access.cypher_builders.native_builders import NativeCypherBuilder
from models.kg_models import CharacterProfile, WorldItem
from tests.test_r08s_state_writer_seams import relationship_row


async def prepare_native_statements(root: Path) -> list[tuple[str, dict[str, Any]]]:
    manager = ContentManager(str(root))
    row = relationship_row()
    reference = manager.save_json([row], "extracted_relationships", "chapter_2", 1)
    loaded = get_extracted_relationships({"project_dir": str(root), "extracted_relationships_ref": reference}, manager)
    relation = ExtractedRelationship(**loaded[0])
    statements = [NativeCypherBuilder.character_upsert_cypher(CharacterProfile(name="Ada", id=row["source_id"], created_chapter=0,
                   personality_description="Protected author profile", motivations="Preserve author intent", physical_description="Protected physical description"), 0),
                  NativeCypherBuilder.world_item_upsert_cypher(WorldItem(name="Named Sword", id=row["target_id"], category="object", created_chapter=0, description="Protected item"), 0)]
    statements.extend(await _build_relationship_statements([relation], [], [], {}, {}, 2, False))
    relationships = {"Bob": [{"type": "ALLIES_WITH", "description": "First", "target_id": "Bob:Exact", "chapter_added": 0, "assertion_origin": "profile"},
                             {"type": "ALLIES_WITH", "description": "Later", "target_id": "Bob:Exact", "chapter_added": 1, "assertion_origin": "profile"},
                             {"type": "TRUSTS", "description": "Trust", "target_id": "Bob:Exact", "chapter_added": 1, "assertion_origin": "profile"}]}
    statements.append(NativeCypherBuilder.character_upsert_cypher(CharacterProfile(name="Reader", id="Reader:Exact", relationships=relationships), 5))
    statements.append(NativeCypherBuilder.world_item_upsert_cypher(WorldItem(name="Place", id="Place:Exact", category="location", relationships={"Other Place": [
        {"type": "BORDERS", "description": "Border", "target_label": "Location", "target_id": "Other:Exact"},
        {"type": "CONTAINS_LOCATION", "description": "Contains", "target_label": "Location", "target_id": "Other:Exact"},
    ]}), 5))
    for name, identifier in [("Mara", "Mara:Exact"), ("mara", "mara:Exact")]:
        statements.append(NativeCypherBuilder.character_upsert_cypher(CharacterProfile(name=name, id=identifier), 0))
    return statements


async def test_native_probe_preparation(tmp_path: Path) -> None:
    import core.langgraph.state as state

    assert Path(state.__file__).resolve() == Path(__file__).resolve().parents[1] / "core/langgraph/state.py"
    statements = await prepare_native_statements(tmp_path)
    assert len(statements) == 8
    assert statements[3][1]["subject_id"] == "Person:Exact-17"
    assert statements[3][1]["object_id"] == "Item:Exact-17"
    assert json.loads(statements[3][1]["relationship_properties"]["scene_assertions"])[1]["scene_index"] == 1
    assert statements[4][1]["relationship_data"][0]["chapter_added"] == 0
    assert len(statements[4][1]["relationship_data"]) == 3
    assert {row["rel_type"] for row in statements[5][1]["relationship_data"]} == {"BORDERS", "CONTAINS_LOCATION"}


async def run_native_readback(root: Path, session: Any) -> dict[str, Any]:
    import config
    from core.entity_embedding_service import build_entity_embedding_update_statements
    from core.service_context import get_services

    assert session.run("MATCH (n) RETURN count(n) AS count").single()["count"] == 0
    statements = await prepare_native_statements(root / "synthetic-content")
    (root / "native-statements.json").write_text(json.dumps(statements, indent=2))
    with session.begin_transaction() as transaction:
        for query, parameters in statements:
            transaction.run(query, parameters).consume()
        transaction.commit()
    row = session.run("MATCH (s:Character {id:'Person:Exact-17'})-[r:WIELDS]->(o:Item {id:'Item:Exact-17'}) RETURN properties(s) AS source, properties(r) AS relationship, properties(o) AS target").single().data()
    assert row["source"]["name"] == "Ada"
    assert row["source"]["created_chapter"] == 0
    assert row["source"]["motivations"] == "Preserve author intent"
    assert row["source"]["personality_description"] == "Protected author profile"
    assert row["source"]["physical_description"] == "Protected physical description"
    assert row["target"]["name"] == "Named Sword"
    assert row["target"]["created_chapter"] == 0
    assert row["relationship"]["scene_index"] == 1
    assert row["relationship"]["chapter_added"] == 2
    assert row["relationship"]["assertion_origin"] == "chapter_extraction"
    assert json.loads(row["relationship"]["scene_assertions"]) == relationship_row()["scene_assertions"]
    before = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
    failures = []
    for changes in [{"source_name": "Other Ada"}, {"source_id": "conflicting"},
                    {"target_type": "Location", "relationship_type": "LOCATED_AT"}]:
        bad = {**relationship_row(), **changes, "scene_assertions": None}
        try:
            batch = await _build_relationship_statements([ExtractedRelationship(**bad)], [], [], {}, {}, 2, False)
            assert len(batch) == 2, "The conflict control must reach the native identity resolver"
            with session.begin_transaction() as transaction:
                for query, parameters in batch:
                    transaction.run(query, parameters).consume()
                transaction.commit()
        except Exception as error:
            if isinstance(error, AssertionError):
                raise
            assert "Explicit canonical identity" in str(error)
            failures.append({"changes": changes, "error_type": type(error).__name__, "error": str(error)})
        else:
            raise AssertionError("Conflicting explicit identity was accepted")
    assert session.run("MATCH (n) RETURN count(n) AS count").single()["count"] == before
    assert session.run("MATCH ()-[r:WIELDS]->() RETURN count(r) AS count").single()["count"] == 1
    projected = session.run("MATCH (c:Character {id:'Reader:Exact'}) OPTIONAL MATCH (c)-[r]->(o) RETURN properties(c) AS c, collect(properties(r) + {type:type(r), target_name:o.name, target_id:o.id, target_label:'Character'}) AS relationships").single().data()
    profile = CharacterProfile.from_dict_record(projected)
    assert len(profile.relationships["Bob"]) == 3
    session.run(*NativeCypherBuilder.character_upsert_cypher(profile, 9)).consume()
    occurrences = session.run("MATCH (:Character {id:'Reader:Exact'})-[r]->() RETURN type(r) AS type, r.chapter_added AS chapter, r.assertion_origin AS origin ORDER BY type, chapter").data()
    assert occurrences == [{"type": "ALLIES_WITH", "chapter": 0, "origin": "profile"}, {"type": "ALLIES_WITH", "chapter": 1, "origin": "profile"}, {"type": "TRUSTS", "chapter": 1, "origin": "profile"}]
    world = session.run("MATCH (:Location {id:'Place:Exact'})-[r]->(:Location {id:'Other:Exact'}) RETURN type(r) AS type ORDER BY type").data()
    assert world == [{"type": "BORDERS"}, {"type": "CONTAINS_LOCATION"}]

    async def read(query: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        return session.run(query, parameters).data()

    async def embed(texts: list[str]) -> list[list[float]]:
        assert texts == ["Mara\nSynthetic description"]
        return [[0.25, 0.75]]

    database = get_services().database
    language_model = get_services().language_model
    with pytest.MonkeyPatch.context() as transport:
        transport.setattr(database, "execute_read_query", read)
        transport.setattr(language_model, "async_get_embeddings_batch", embed)
        with config.bind_settings(config.snapshot_settings().model_copy(update={"EXPECTED_EMBEDDING_DIM": 2, "NEO4J_VECTOR_DIMENSIONS": 2, "ENABLE_ENTITY_EMBEDDING_PERSISTENCE": True})):
            vectors = await build_entity_embedding_update_statements(characters=[CharacterProfile(name="Mara", personality_description="Synthetic description")], world_items=[])
            for query, parameters in vectors:
                session.run(query, parameters).consume()
            property_name = config.ENTITY_EMBEDDING_VECTOR_PROPERTY
            vector_rows = session.run(f"MATCH (c:Character) WHERE c.id IN ['Mara:Exact', 'mara:Exact'] RETURN c.id AS id, c.`{property_name}` AS vector ORDER BY id").data()
            assert vector_rows == [{"id": "Mara:Exact", "vector": [0.25, 0.75]}, {"id": "mara:Exact", "vector": None}]
    return {"status": "passed", "relationship_readback": row, "profile_occurrences": occurrences, "world_predicates": world, "conflicts_rejected": failures, "vectors": vector_rows}


if __name__ == "__main__":
    # The coordinator's isolated engine runner calls this module, never normal pytest.
    from neo4j import GraphDatabase

    root = Path(os.environ["R08S_PROBE_ROOT"])
    assert root.is_relative_to(Path(os.environ["HOME"]))
    with GraphDatabase.driver("bolt://127.0.0.1:17831", auth=("neo4j", "synthetic-r08s-password"), max_transaction_retry_time=0) as driver:
        with driver.session(database="neo4j") as session:
            result = asyncio.run(run_native_readback(root, session))
    (root / "readback.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
