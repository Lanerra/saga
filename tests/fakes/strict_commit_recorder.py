"""Exact commit submission contracts over a seeded, read-only synthetic catalog.

Writes are recorded, not applied. Use the journal-capable DriverExample for
transaction/compensation assertions and Neo4j for real persistence evidence.
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

import config
from data_access.chapter_queries import build_chapter_upsert_statement
from data_access.cypher_builders.native_builders import NativeCypherBuilder, chapter_assertion_delete_statement, relationship_statement
from models.kg_models import CharacterProfile, WorldItem
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager, QueryResponse


class StrictCommitRecorder(FakeNeo4jManager):
    """Declare only complete native commit signatures; reject other operations."""

    def __init__(self, *, nodes: list[dict[str, Any]] | None = None) -> None:
        super().__init__()
        self.nodes = deepcopy(nodes) if nodes is not None else []
        self.text_hash_property = config.ENTITY_EMBEDDING_TEXT_HASH_PROPERTY
        self.model_property = config.ENTITY_EMBEDDING_MODEL_PROPERTY
        self.vector_property = config.ENTITY_EMBEDDING_VECTOR_PROPERTY
        match_candidates = """
            OPTIONAL MATCH (candidate)
            WHERE entity.label IN labels(candidate)
              AND CASE WHEN entity.id IS NOT NULL THEN candidate.id = entity.id
                       ELSE candidate.name = entity.name END
            WITH entity, collect(candidate) AS candidates
        """
        self.embedding_lookup_query = f"""
            UNWIND $entities AS entity
            {match_candidates}
            CALL apoc.util.validate(size(candidates) > 1, 'Ambiguous canonical entity', [])
            WITH entity, head(candidates) AS found
            CALL apoc.util.validate(found IS NOT NULL AND (found.id IS NULL OR trim(found.id) = ''),
                                    'Canonical entity has no stable ID', [])
            RETURN entity.index AS key, found.id AS id, found.`{self.text_hash_property}` AS existing_hash,
                   found.`{self.model_property}` AS existing_model,
                   found.`{self.model_property}_identity` AS existing_identity,
                   found.`{self.vector_property}` AS existing_vector
        """
        self.embedding_update_query = f"""
            WITH $identity AS entity
            {match_candidates}
            CALL apoc.util.validate(size(candidates) <> 1, 'Embedding target must resolve exactly once', [])
            WITH head(candidates) AS node
            CALL apoc.util.validate(node.id IS NULL OR trim(node.id) = '', 'Canonical entity has no stable ID', [])
            SET node.`{self.vector_property}` = $vector,
                node.`{self.text_hash_property}` = $text_hash,
                node.`{self.model_property}` = $model,
                node.`{self.model_property}_identity` = $embedding_identity,
                node.updated_ts = timestamp()
        """
        self.register_exact(self.embedding_lookup_query, self._embedding_rows)
        self.register_exact("""
            MATCH (n)
            WHERE n:Character OR n:Location OR n:Event OR n:Item
            RETURN DISTINCT toLower(n.name) AS name
        """, self._entity_names)
        self._register_write(self.embedding_update_query, {"identity", "vector", "text_hash", "model", "embedding_identity"})
        templates = [
            chapter_assertion_delete_statement(1),
            NativeCypherBuilder.character_upsert_cypher(CharacterProfile(name="Synthetic Character"), 1, assertion_origin="chapter_profile"),
            NativeCypherBuilder.world_item_upsert_cypher(WorldItem.from_dict("Location", "Synthetic Location", {}), 1, assertion_origin="chapter_profile"),
            relationship_statement({"type": "Character", "name": "Synthetic Source"}, "KNOWS", {"type": "Character", "name": "Synthetic Target"},
                                   1, origin="chapter_extraction", provisional=False),
            build_chapter_upsert_statement(chapter_number=1),
        ]
        for query, parameters in templates:
            self._register_write(query, set(parameters))

    def register_exact(self, query: str, response: QueryResponse) -> None:
        # Whitespace varies with nested Cypher fragments; every token remains literal.
        pattern = r"\A\s*" + r"\s+".join(re.escape(token) for token in query.split()) + r"\s*\Z"
        self.configure_response(f"(?-i:{pattern})", response)

    def _register_write(self, query: str, keys: set[str]) -> None:
        def acknowledge(parameters: dict[str, Any] | None) -> list[dict[str, Any]]:
            assert parameters is not None and set(parameters) == keys, "Unexpected native write parameters"
            return []

        self.register_exact(query, acknowledge)

    def _entity_names(self, parameters: dict[str, Any] | None) -> list[dict[str, Any]]:
        assert parameters == {}, "Unexpected entity-name parameters"
        return [{"name": name} for name in sorted({node["properties"]["name"].lower() for node in self.nodes
                                                   if set(node["labels"]) & {"Character", "Location", "Item", "Event"}})]

    def _embedding_rows(self, parameters: dict[str, Any] | None) -> list[dict[str, Any]]:
        assert parameters is not None and set(parameters) == {"entities"}, "Unexpected embedding lookup parameters"
        rows = []
        for entity in parameters["entities"]:
            assert set(entity) == {"index", "label", "id", "name", "category", "description"}, "Unexpected embedding candidate fields"
            candidates = [node for node in self.nodes if entity["label"] in node["labels"] and (
                node["properties"].get("id") == entity["id"] if entity["id"] is not None else node["properties"].get("name") == entity["name"]
            )]
            assert len(candidates) <= 1, "Ambiguous canonical entity"
            stored = candidates[0]["properties"] if candidates else {}
            if candidates:
                assert isinstance(stored.get("id"), str) and stored["id"].strip(), "Canonical entity has no stable ID"
            rows.append({
                "key": entity["index"], "id": stored.get("id"), "existing_hash": stored.get(self.text_hash_property),
                "existing_model": stored.get(self.model_property), "existing_identity": stored.get(f"{self.model_property}_identity"),
                "existing_vector": deepcopy(stored.get(self.vector_property)),
            })
        return rows
