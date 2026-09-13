"""Production enrichment callers and complete chapter vector admission."""
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from neo4j import Transaction
from pydantic import ValidationError

import config
from core.embedding_contract import embedding_identity
from core.exceptions import DatabaseError
from core.langgraph.nodes import narrative_enrichment_node as node
from core.parsers.narrative_enrichment_parser import ChapterEmbeddingExtractionResult, NarrativeEnrichmentParser
from core.service_context import get_services
from data_access import chapter_queries
from models.kg_models import CharacterProfile
from tests.test_langgraph.test_chapter_lifecycle import Rows

pytestmark = pytest.mark.run_settings(EXPECTED_EMBEDDING_DIM=3, NEO4J_VECTOR_DIMENSIONS=3, EMBEDDING_DTYPE="float32", ENABLE_CHAPTER_EMBEDDING_EXTRACTION=True, ENABLE_PHYSICAL_DESCRIPTION_EXTRACTION=False)

@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Path(chapter_queries.__file__).resolve().parents[1] == Path(__file__).resolve().parents[1]
    assert config.snapshot_settings().EXPECTED_EMBEDDING_DIM == 3


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['metadata_only', 'current', 'stale', 'missing_model', 'missing_identity', 'wrong_identity', 'nan', 'infinity', 'wrong_dimensions', 'nested', 'empty'])
async def test_complete_chapter_reader_admission(configured: None, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    vector: Any = [1.0, 0.0, 0.0]
    if kind == 'nan':
        vector = [float('nan'), 0.0, 0.0]
    if kind == 'wrong_dimensions':
        vector = [1.0, 0.0]
    if kind == 'metadata_only':
        vector = None
    if kind == 'infinity':
        vector = [float('inf'), 0.0, 0.0]
    if kind == 'nested':
        vector = [[1.0, 0.0, 0.0]]
    if kind == 'empty':
        vector = []
    row = {'id': 'synthetic-chapter', 'number': 1, 'title': 'Synthetic',
           'act_number': 1, 'embedding': vector,
           'embedding_model': 'stale' if kind == 'stale' else config.EMBEDDING_MODEL,
           'embedding_identity': None if kind == 'missing_identity' else embedding_identity()}
    if kind == 'missing_model':
        row['embedding_model'] = None
    if kind == 'wrong_identity':
        row['embedding_identity'] = 'stale-fingerprint'
    async def read(query: str, parameters: Any = None) -> list[dict[str, Any]]:
        return [row]
    monkeypatch.setattr(get_services().database, 'execute_read_query', read)
    if kind not in ('current', 'metadata_only'):
        with pytest.raises(DatabaseError):
            await chapter_queries.get_chapter_data_from_db(1)
        return
    chapter = await chapter_queries.get_chapter_data_from_db(1)
    assert chapter is not None
    assert chapter.embedding == vector


@pytest.mark.asyncio
@pytest.mark.parametrize('entrypoint', ['node', 'parser'])
@pytest.mark.parametrize('existing', [False, True])
async def test_existing_enrichment_caller_reaches_real_writer(configured: None, monkeypatch: pytest.MonkeyPatch, entrypoint: str, existing: bool) -> None:
    writes = []
    async def read(query: str, parameters: Any = None) -> list[dict[str, Any]]:
        return [{'id': 'synthetic-chapter', 'number': 1, 'title': 'Synthetic', 'act_number': 1,
                 'embedding': [1.0, 0.0, 0.0] if existing else None,
                 'embedding_model': config.EMBEDDING_MODEL, 'embedding_identity': embedding_identity()}]
    async def write(query: str, parameters: Any = None) -> None:
        writes.append((query, parameters))
    async def profiles() -> list[CharacterProfile]:
        return [CharacterProfile(name='Synthetic', id='synthetic-character')]
    async def embed(text: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0])
    monkeypatch.setattr(get_services().database, 'execute_read_query', read)
    monkeypatch.setattr(get_services().database, 'execute_write_query', write)
    monkeypatch.setattr(get_services().language_model, 'async_get_embedding', embed)
    monkeypatch.setattr(node, 'get_character_profiles', profiles)
    if entrypoint == 'node':
        candidate = await node.NarrativeEnrichmentNode().process('Synthetic narrative.', 1)
        assert writes == []

        class CandidateTransaction:
            def run(self, query: str, parameters: Any = None) -> Rows:
                if "RETURN c.embedding_vector AS embedding" in query:
                    return Rows([{"embedding": [1.0, 0.0, 0.0] if existing else None}])
                writes.append((query, parameters))
                return Rows()

        candidate.apply(cast(Transaction, CandidateTransaction()), 1)
    else:
        result, message = await NarrativeEnrichmentParser('Synthetic narrative.', 1).parse_and_persist()
        assert result, message
    assert len(writes) == 1
    parameters = writes[0][1]
    assert parameters['embedding_vector_param'] == [1.0, 0.0, 0.0]
    assert parameters['embedding_model_param'] == config.EMBEDDING_MODEL
    assert parameters['embedding_identity_param'] == embedding_identity()


def test_result_rejects_boolean_coercion(configured: None) -> None:
    with pytest.raises(ValidationError):
        ChapterEmbeddingExtractionResult(
            chapter_number=1, embedding_vector=[True, 0.0, 0.0],
            embedding_model=config.EMBEDDING_MODEL, embedding_identity=embedding_identity(),
        )


@pytest.mark.parametrize("contract_change", ["model", "endpoint"])
async def test_extracted_identity_cannot_cross_runs(configured: None, monkeypatch: pytest.MonkeyPatch, contract_change: str) -> None:
    async def read(query: str, parameters: Any = None) -> list[dict[str, Any]]:
        return [{"id": "synthetic", "number": 1, "title": "Synthetic", "act_number": 1}]

    async def embed(text: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0])

    writes = []

    async def write(query: str, parameters: Any = None) -> None:
        writes.append(parameters)

    monkeypatch.setattr(get_services().database, "execute_read_query", read)
    monkeypatch.setattr(get_services().database, "execute_write_query", write)
    monkeypatch.setattr(get_services().language_model, "async_get_embedding", embed)
    parser = NarrativeEnrichmentParser("Synthetic narrative.", 1)
    results = await parser.extract_chapter_embeddings()
    assert len(results) == 1
    assert results[0].embedding_model == config.EMBEDDING_MODEL
    assert results[0].embedding_identity == embedding_identity()
    override = {"EMBEDDING_MODEL": "another-synthetic-model"} if contract_change == "model" else {"EMBEDDING_API_BASE": "http://synthetic.invalid:2"}
    effective = config.EffectiveSettings.model_validate({**config.snapshot_settings().model_dump(), **override})
    with config.bind_settings(effective):
        assert await parser.update_chapter_embeddings(results) is False
    assert writes == []


@pytest.mark.parametrize("vector", [[float("nan"), 0.0, 0.0], [[1.0, 0.0, 0.0]], [1.0, 0.0], [True, 0.0, 0.0]])
async def test_extraction_omits_invalid_provider_vectors(configured: None, monkeypatch: pytest.MonkeyPatch, vector: Any) -> None:
    async def read(query: str, parameters: Any = None) -> list[dict[str, Any]]:
        return [{"id": "synthetic", "number": 1, "title": "Synthetic", "act_number": 1}]

    async def embed(text: str) -> Any:
        return vector

    monkeypatch.setattr(get_services().database, "execute_read_query", read)
    monkeypatch.setattr(get_services().language_model, "async_get_embedding", embed)
    assert await NarrativeEnrichmentParser("Synthetic narrative.", 1).extract_chapter_embeddings() == []
