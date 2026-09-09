"""Stored vectors cannot influence healing across producer contracts."""
from typing import Any

import pytest

import config
from core.embedding_contract import embedding_identity
from core.graph_healing_service import GraphHealingService
from core.service_context import get_services


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['valid', 'stale_model', 'stale_identity', 'nan', 'rank', 'dimensions'])
async def test_healing_admits_only_current_finite_vectors(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, 'EXPECTED_EMBEDDING_DIM', 3)
    monkeypatch.setattr(config, 'ENABLE_ENTITY_EMBEDDING_GRAPH_HEALING', True)

    async def candidates(**kwargs: Any) -> list[dict[str, Any]]:
        return [{'id1': 'a', 'id2': 'b', 'name1': 'A', 'name2': 'B', 'similarity': 0.5, 'labels1': ['Character']}]

    async def read(query: str, parameters: Any = None) -> list[dict[str, Any]]:
        if 'AS embedding_vector' in query:
            vector: Any = [1.0, 0.0, 0.0]
            if kind == 'nan':
                vector = [float('nan'), 0.0, 0.0]
            elif kind == 'rank':
                vector = [vector]
            elif kind == 'dimensions':
                vector = [1.0, 0.0]
            return [{'id': key, 'embedding_vector': vector,
                     'embedding_model': 'stale' if kind == 'stale_model' else config.EMBEDDING_MODEL,
                     'embedding_identity': 'stale' if kind == 'stale_identity' else embedding_identity()} for key in ['a', 'b']]
        return [{'entity_id': key, 'element_id': key} for key in ['a', 'b']]

    monkeypatch.setattr('data_access.kg_queries.find_candidate_duplicate_entities', candidates)
    monkeypatch.setattr(get_services().database, 'execute_read_query', read)
    result = await GraphHealingService().find_merge_candidates()
    assert len(result) == 1
    assert result[0]['embedding_similarity'] == (1.0 if kind == 'valid' else 0.0)
    assert result[0]['similarity'] == (0.8 if kind == 'valid' else 0.5)
