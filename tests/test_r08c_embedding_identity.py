"""Emitted-query identity contract; not a Neo4j engine verification."""

from typing import Any

import pytest

from core.entity_embedding_service import build_entity_embedding_update_statements
from core.service_context import get_services
from models.kg_models import CharacterProfile


@pytest.mark.run_settings(EXPECTED_EMBEDDING_DIM=2, NEO4J_VECTOR_DIMENSIONS=2, ENABLE_ENTITY_EMBEDDING_PERSISTENCE=True)
@pytest.mark.parametrize("identifier", ["", "literal-ID"])
async def test_embedding_queries_preserve_literal_name_identity(identifier: str, monkeypatch: pytest.MonkeyPatch) -> None:
    reads: list[tuple[str, dict[str, Any]]] = []

    async def read(query: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        assert "UNWIND $entities AS entity" in query
        assert "RETURN entity.index AS key" in query
        assert set(parameters) == {"entities"}
        reads.append((query, parameters))
        return [{"key": 0, "id": identifier or None, "existing_hash": None}]

    async def embed(texts: list[str]) -> list[list[float]]:
        assert texts == ["Mara\nSynthetic description"]
        return [[0.25, 0.75]]

    monkeypatch.setattr(get_services().database, "execute_read_query", read)
    monkeypatch.setattr(get_services().language_model, "async_get_embeddings_batch", embed)
    statements = await build_entity_embedding_update_statements(characters=[CharacterProfile(name="Mara", id=identifier, personality_description="Synthetic description")], world_items=[])
    assert len(reads) == len(statements) == 1
    assert reads[0][1]["entities"][0]["name"] == "Mara"
    assert reads[0][1]["entities"][0]["id"] == (identifier or None)
    assert statements[0][1]["identity"] == {"label": "Character", "id": identifier or None, "name": "Mara"}
    for query in (reads[0][0], statements[0][0]):
        assert "candidate.id = entity.id" in query
        if not identifier:
            assert "toLower(trim(candidate.name))" not in query, "Embedding resolver must not merge case-distinct literal names"
            assert "candidate.name = entity.name" in query
