"""Storage and query admission with disposable content and narrow I/O fakes."""
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import config
from core.entity_embedding_service import build_entity_embedding_update_statements, compute_entity_embedding_text_hash
from core.langgraph.content_manager import ContentManager, load_scene_embeddings, save_scene_embeddings
from core.langgraph.nodes.commit_graph_ops import _aggregate_scene_embeddings_to_chapter
from core.llm_interface_refactored import EmbeddingService
from core.service_context import get_services
from data_access.chapter_queries import build_chapter_upsert_statement, find_semantic_context_native
from models.kg_models import CharacterProfile


class EmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def get_embedding(self, text: str, model: str) -> dict[str, Any]:
        self.calls.append((text, model))
        return {'embedding': [float(len(self.calls)), 0.25, 0.5]}


@pytest.fixture
def vector_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, 'EXPECTED_EMBEDDING_DIM', 3)
    monkeypatch.setattr(config, 'NEO4J_VECTOR_DIMENSIONS', 3)
    monkeypatch.setattr(config, 'EMBEDDING_DTYPE', 'float32')
    monkeypatch.setattr(config, 'EMBEDDING_MODEL', 'synthetic-vector-a')


@pytest.mark.asyncio
@pytest.mark.parametrize('field,value', [('EMBEDDING_MODEL', 'synthetic-vector-b'), ('EMBEDDING_API_BASE', 'https://different.invalid'), ('EMBEDDING_MAX_INPUT_TOKENS', 3), ('EMBEDDING_DTYPE', 'float64')])
async def test_cache_identity_tracks_producer_contract(vector_configuration: None, monkeypatch: pytest.MonkeyPatch, field: str, value: Any) -> None:
    from core.lightweight_cache import clear_service_cache
    clear_service_cache('llm_embedding')
    provider = EmbeddingProvider()
    first = EmbeddingService(provider)  # type: ignore[arg-type]
    original = await first.get_embedding('same synthetic text')
    monkeypatch.setattr(config, field, value)
    second = EmbeddingService(provider)  # type: ignore[arg-type]
    changed = await second.get_embedding('same synthetic text')
    assert original is not None and changed is not None
    assert original.tolist() == [1.0, 0.25, 0.5]
    assert changed.tolist() == [2.0, 0.25, 0.5]
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_returned_array_cannot_poison_cache(vector_configuration: None) -> None:
    from core.lightweight_cache import clear_service_cache
    clear_service_cache('llm_embedding')
    provider = EmbeddingProvider()
    service = EmbeddingService(provider)  # type: ignore[arg-type]
    original = await service.get_embedding('mutation synthetic')
    assert original is not None
    original[0] = float('nan')
    cached = await service.get_embedding('mutation synthetic')
    assert cached is not None
    assert cached.tolist() == [1.0, 0.25, 0.5]
    assert len(provider.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('vector', [[[1.0, 2.0, 3.0]], [float('nan'), 1.0, 2.0], [float('inf'), 1.0, 2.0], [1.0, 2.0], [True, 1.0, 2.0]])
async def test_entity_direct_output_rejected_before_statement(vector_configuration: None, monkeypatch: pytest.MonkeyPatch, vector: Any) -> None:
    async def read(query: str, parameters: Any = None) -> list[Any]:
        return []

    async def embed(texts: list[str]) -> list[Any]:
        return [vector]

    monkeypatch.setattr(get_services().database, 'execute_read_query', read)
    monkeypatch.setattr(get_services().language_model, 'async_get_embeddings_batch', embed)
    with pytest.raises((TypeError, ValueError)):
        await build_entity_embedding_update_statements(characters=[CharacterProfile(name='Synthetic', id='synthetic-id')], world_items=[])


@pytest.mark.asyncio
async def test_same_text_different_persisted_model_requires_regeneration(vector_configuration: None, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    async def read(query: str, parameters: Any = None) -> list[dict[str, Any]]:
        return [{'key': 0, 'id': 'synthetic-id', 'existing_hash': compute_entity_embedding_text_hash('Synthetic'), 'existing_model': 'stale-model'}]

    async def embed(texts: list[str]) -> list[Any]:
        calls.append(texts)
        return [np.array([1.0, 2.0, 3.0])]

    monkeypatch.setattr(get_services().database, 'execute_read_query', read)
    monkeypatch.setattr(get_services().language_model, 'async_get_embeddings_batch', embed)
    statements = await build_entity_embedding_update_statements(characters=[CharacterProfile(name='Synthetic', id='synthetic-id')], world_items=[])
    assert calls == [['Synthetic']]
    assert statements[0][1]['model'] == 'synthetic-vector-a'
    assert statements[0][1]['vector'] == [1.0, 2.0, 3.0]


@pytest.mark.parametrize('vector', [[[1.0, 2.0, 3.0]], [float('nan'), 1.0, 2.0], [float('inf'), 1.0, 2.0], [1.0, 2.0]])
def test_chapter_writer_and_aggregation_reject_malformed_vectors(vector_configuration: None, vector: Any) -> None:
    with pytest.raises(ValueError):
        build_chapter_upsert_statement(chapter_number=1, embedding_vector=vector, embedding_model=config.EMBEDDING_MODEL)
    with pytest.raises(ValueError):
        _aggregate_scene_embeddings_to_chapter([vector])


@pytest.mark.asyncio
async def test_query_rejects_malformed_direct_vector_before_database(vector_configuration: None, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def read(query: str, parameters: Any = None) -> list[Any]:
        calls.append(query)
        return []

    monkeypatch.setattr(get_services().database, 'execute_read_query', read)
    with pytest.raises(ValueError):
        await find_semantic_context_native(np.array([float('nan'), 1.0, 2.0]), 2, embedding_model=config.EMBEDDING_MODEL)
    assert calls == []


def test_scene_artifact_cannot_cross_model(vector_configuration: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))
    reference = save_scene_embeddings(manager, [[1.0, 2.0, 3.0]], 1, embedding_model=config.EMBEDDING_MODEL)
    assert load_scene_embeddings(manager, reference) == [[1.0, 2.0, 3.0]]
    monkeypatch.setattr(config, 'EMBEDDING_MODEL', 'synthetic-vector-b')
    with pytest.raises(ValueError, match='identity|model'):
        load_scene_embeddings(manager, reference)


@pytest.mark.asyncio
async def test_commit_refuses_unidentified_legacy_vector(vector_configuration: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from core.langgraph.nodes.commit_node import commit_to_graph
    from core.langgraph.state import NarrativeState
    manager = ContentManager(str(tmp_path))
    state = cast(NarrativeState, {'project_dir': str(tmp_path), 'current_chapter': 1,
                            'draft_ref': manager.save_text('Synthetic draft.', 'draft', '1'),
                            'generated_embedding': [1.0, 2.0, 3.0]})
    batches: list[Any] = []

    async def read(query: str, parameters: Any = None) -> list[Any]:
        return []

    async def batch(statements: Any) -> None:
        batches.append(statements)

    monkeypatch.setattr(get_services().database, 'execute_read_query', read)
    monkeypatch.setattr(get_services().database, 'execute_cypher_batch', batch)
    result = await commit_to_graph(state)
    assert result['has_fatal_error'] is True
    assert isinstance(result['last_error'], str)
    assert 'identity' in result['last_error']
    assert batches == []
