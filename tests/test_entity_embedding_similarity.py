# tests/test_entity_embedding_similarity.py
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from core.graph_healing_service import GraphHealingService
from processing.entity_deduplication import check_entity_similarity


@pytest.mark.asyncio
async def test_check_entity_similarity_world_item_uses_embedding_vector_search_when_enabled() -> None:
    async def _execute_read_query(query: str, parameters=None):
        if "CALL db.index.vector.queryNodes" in query:
            return [
                {
                    "existing_id": "locations_castle",
                    "existing_name": "Castle",
                    "existing_category": "Locations",
                    "existing_labels": ["Location"],
                    "existing_description": "A stone fortress.",
                    "similarity": 0.91,
                }
            ]
        raise AssertionError(f"Unexpected query in test: {query!r}")

    with (
        patch("config.ENABLE_ENTITY_EMBEDDING_DEDUPLICATION", True),
        patch("config.ENTITY_EMBEDDING_DEDUPLICATION_TOP_K", 5),
        patch("config.ENTITY_EMBEDDING_DEDUPLICATION_SIMILARITY_THRESHOLD", 0.9),
        patch(
            "core.llm_interface_refactored.llm_service.async_get_embedding",
            new=AsyncMock(return_value=np.array([1.0, 0.0], dtype=np.float32)),
        ),
        patch(
            "processing.entity_deduplication.neo4j_manager.execute_read_query",
            new=AsyncMock(side_effect=_execute_read_query),
        ),
    ):
        result = await check_entity_similarity(
            "Castle",
            "world_element",
            "Locations",
            description="A stone fortress.",
        )

    assert result is not None
    assert result["existing_id"] == "locations_castle"
    assert result["existing_name"] == "Castle"
    assert result["similarity_source"] == "embedding"
    assert result["similarity"] == pytest.approx(0.91)


@pytest.mark.asyncio
async def test_graph_healing_advanced_matching_includes_embedding_similarity_when_enabled() -> None:
    service = GraphHealingService()

    kg_candidates = [
        {
            "id1": "entity_1",
            "name1": "Alpha",
            "labels1": ["Character"],
            "id2": "entity_2",
            "name2": "Alfa",
            "labels2": ["Character"],
            "similarity": 0.8,
        }
    ]

    async def _execute_read_query(query: str, parameters=None):
        if "WHERE n.id IN $ids" in query and "embedding_vector" in query:
            return [
                {"id": "entity_1", "labels": ["Character"], "embedding_vector": [1.0, 0.0]},
                {"id": "entity_2", "labels": ["Character"], "embedding_vector": [1.0, 0.0]},
            ]

        if "RETURN elementId(n) AS element_id" in query:
            entity_id = parameters.get("entity_id") if isinstance(parameters, dict) else None
            if entity_id == "entity_1":
                return [{"element_id": "element-1"}]
            if entity_id == "entity_2":
                return [{"element_id": "element-2"}]
            return []

        raise AssertionError(f"Unexpected query in test: {query!r}")

    with (
        patch(
            "data_access.kg_queries.find_candidate_duplicate_entities",
            new=AsyncMock(return_value=kg_candidates),
        ),
        patch(
            "core.graph_healing_service.neo4j_manager.execute_read_query",
            new=AsyncMock(side_effect=_execute_read_query),
        ),
        patch("core.graph_healing_service.config.ENABLE_ENTITY_EMBEDDING_GRAPH_HEALING", True),
    ):
        candidates = await service.find_merge_candidates(use_advanced_matching=True)

    assert len(candidates) == 1
    c = candidates[0]
    assert c["primary_id"] == "element-1"
    assert c["duplicate_id"] == "element-2"
    assert c["name_similarity"] == pytest.approx(0.8)
    assert c["embedding_similarity"] == pytest.approx(1.0)

    # combined = 0.4*name + 0.6*emb
    assert c["similarity"] == pytest.approx(0.92)
