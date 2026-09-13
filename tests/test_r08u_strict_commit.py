"""Response recorder contracts, not Neo4j persistence evidence."""
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import config
from core.embedding_contract import embedding_identity
from core.entity_embedding_service import build_entity_embedding_update_statements, compute_entity_embedding_text_hash
from core.langgraph.nodes.commit_node import _get_existing_entity_names
from core.service_context import get_services
from data_access.chapter_queries import build_chapter_upsert_statement
from data_access.cypher_builders.native_builders import NativeCypherBuilder, chapter_assertion_delete_statement, relationship_statement
from models.kg_models import CharacterProfile, WorldItem
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager
from tests.fakes.strict_commit_recorder import StrictCommitRecorder


@pytest.mark.parametrize("operation", ["execute_read_query", "execute_write_query", "execute_cypher_batch"])
async def test_parameter_callbacks_record_exact_inputs(operation: str) -> None:
    database = FakeNeo4jManager()
    calls: list[dict[str, Any] | None] = []

    def respond(parameters: dict[str, Any] | None) -> list[dict[str, Any]]:
        calls.append(parameters)
        assert parameters is not None
        return [{"name": parameters["name"], "id": parameters["id"]}]

    query = "RETURN $name AS name, $id AS id"
    database.configure_response(r"\ARETURN \$name AS name, \$id AS id\Z", respond)
    parameters = {"name": "Exact Name", "id": "literal-ID:01"}
    if operation == "execute_cypher_batch":
        await database.execute_cypher_batch([(query, parameters)])
        assert database.batch_statements == [[(query, parameters)]]
    else:
        assert await getattr(database, operation)(query, parameters) == [parameters]
    assert calls == [parameters]
    assert database.executed_queries == [(query, parameters)]
    with pytest.raises(AssertionError, match="Unconfigured synthetic query"):
        await database.execute_read_query(query + " RETURN 2", parameters)
    assert calls == [parameters]


async def test_callback_none_exception_and_reset_contract() -> None:
    database = FakeNeo4jManager()
    calls: list[dict[str, Any] | None] = []

    def respond(parameters: dict[str, Any] | None) -> list[dict[str, Any]]:
        calls.append(parameters)
        raise ValueError("Synthetic callback failure")

    database.configure_response(r"\ARETURN 1\Z", respond)
    with pytest.raises(ValueError, match="Synthetic callback failure"):
        await database.execute_read_query("RETURN 1")
    assert calls == [None]
    database.reset()
    assert database.executed_queries == []
    with pytest.raises(AssertionError, match="Unconfigured"):
        await database.execute_read_query("RETURN 1")
    assert calls == [None]


def candidate(index: int, label: str, name: str, identifier: str | None = None) -> dict[str, Any]:
    return {"index": index, "label": label, "name": name, "id": identifier, "category": "", "description": ""}


async def test_lookup_returns_seeded_fields_per_input_and_exact_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    fields = {
        "name": "Alice", "id": "person:Literal-ID",
        config.ENTITY_EMBEDDING_TEXT_HASH_PROPERTY: "stored-hash",
        config.ENTITY_EMBEDDING_MODEL_PROPERTY: "stored-model",
        f"{config.ENTITY_EMBEDDING_MODEL_PROPERTY}_identity": "stored-producer",
        config.ENTITY_EMBEDDING_VECTOR_PROPERTY: [0.25, 0.75],
    }
    nodes = [
        {"labels": ["Character"], "properties": fields},
        {"labels": ["Location"], "properties": {"name": "Alice", "id": "place:Literal-ID"}},
    ]
    database = StrictCommitRecorder(nodes=nodes)
    before = deepcopy(database.nodes)
    parameters = {"entities": [
        candidate(8, "Character", "Different Name", "person:Literal-ID"),
        candidate(2, "Location", "Alice"),
        candidate(5, "Character", "Alice", "unknown-ID"),
        candidate(3, "Character", "alice"),
        candidate(4, "Item", "Alice"),
    ]}
    rows = await database.execute_read_query(database.embedding_lookup_query, parameters)
    empty_fields = {"existing_hash": None, "existing_model": None, "existing_identity": None, "existing_vector": None}
    assert rows == [
        {"key": 8, "id": "person:Literal-ID", "existing_hash": "stored-hash", "existing_model": "stored-model",
         "existing_identity": "stored-producer", "existing_vector": [0.25, 0.75]},
        {"key": 2, "id": "place:Literal-ID", **empty_fields},
        *[{"key": index, "id": None, **empty_fields} for index in (5, 3, 4)],
    ]
    rows[0]["existing_vector"][0] = 99
    assert database.nodes == before == nodes
    monkeypatch.setattr(get_services(), "database", database)
    assert await _get_existing_entity_names() == {"alice"}
    assert database.batch_statements == []


@pytest.mark.parametrize("identifier", [None, "", "   "])
async def test_seeded_entity_without_stable_id_is_rejected(identifier: str | None) -> None:
    database = StrictCommitRecorder(nodes=[{"labels": ["Character"], "properties": {"name": "Alice", "id": identifier}}])
    with pytest.raises(AssertionError, match="Canonical entity has no stable ID"):
        await database.execute_read_query(database.embedding_lookup_query, {"entities": [candidate(0, "Character", "Alice")]})


async def test_ambiguous_seeded_identity_is_rejected() -> None:
    database = StrictCommitRecorder(nodes=[{"labels": ["Character"], "properties": {"name": "Alice", "id": identifier}} for identifier in ("one", "two")])
    with pytest.raises(AssertionError, match="Ambiguous canonical entity"):
        await database.execute_read_query(database.embedding_lookup_query, {"entities": [candidate(0, "Character", "Alice")]})


@pytest.mark.run_settings(EXPECTED_EMBEDDING_DIM=2, EMBEDDING_DTYPE="float64", ENTITY_EMBEDDING_TEXT_HASH_PROPERTY="fixture_hash",
                         ENTITY_EMBEDDING_MODEL_PROPERTY="fixture_model", ENTITY_EMBEDDING_VECTOR_PROPERTY="fixture_vector")
async def test_native_embedding_builder_reuses_seeded_producer_and_preserves_new_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    database = StrictCommitRecorder(nodes=[{"labels": ["Character"], "properties": {
        "name": "Alice", "id": "person:Alice", "fixture_hash": compute_entity_embedding_text_hash("Alice\nScout"),
        "fixture_model": config.EMBEDDING_MODEL, "fixture_model_identity": embedding_identity(), "fixture_vector": [0.25, 0.75],
    }}])
    before = deepcopy(database.nodes)
    monkeypatch.setattr(get_services(), "database", database)
    calls = []

    async def embeddings(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[0.5, 0.5] for _ in texts]

    monkeypatch.setattr(get_services().language_model, "async_get_embeddings_batch", embeddings)
    statements = await build_entity_embedding_update_statements(
        characters=[CharacterProfile(name="Alice", personality_description="Scout"), CharacterProfile(id="person:Bob", name="Bob", personality_description="Guide")],
        world_items=[WorldItem.from_dict("Location", "Hall", {"id": "place:Hall", "description": "Wide"})],
    )
    assert calls == [["Bob\nGuide", "Hall\nLocation\nWide"]]
    assert len(statements) == 2
    assert [parameters["identity"] for _, parameters in statements] == [
        {"label": "Character", "id": "person:Bob", "name": "Bob"}, {"label": "Location", "id": "place:Hall", "name": "Hall"},
    ]
    assert [parameters["vector"] for _, parameters in statements] == [[0.5, 0.5], [0.5, 0.5]]
    assert all(parameters["embedding_identity"] == embedding_identity() for _, parameters in statements)
    await database.execute_cypher_batch(statements)
    assert database.batch_statements == [statements]
    assert database.nodes == before
    assert Path(build_entity_embedding_update_statements.__code__.co_filename).resolve() == Path(__file__).resolve().parents[1] / "core/entity_embedding_service.py"


@pytest.mark.parametrize("replacement", ["found.id AS wrong_id", "found.`wrong_hash` AS existing_hash", "found.`wrong_model` AS existing_model",
                                         "found.`wrong_identity` AS existing_identity", "found.`wrong_vector` AS existing_vector"])
async def test_lookup_rejects_changed_complete_signature(replacement: str) -> None:
    database = StrictCommitRecorder()
    original = {
        "wrong_id": "found.id AS id", "existing_hash": f"found.`{database.text_hash_property}` AS existing_hash",
        "existing_model": f"found.`{database.model_property}` AS existing_model", "existing_identity": f"found.`{database.model_property}_identity` AS existing_identity",
        "existing_vector": f"found.`{database.vector_property}` AS existing_vector",
    }[replacement.split()[-1]]
    with pytest.raises(AssertionError, match="Unconfigured synthetic query"):
        await database.execute_read_query(database.embedding_lookup_query.replace(original, replacement), {"entities": []})


async def test_known_native_batch_records_order_values_and_rejects_unknown_writes() -> None:
    database = StrictCommitRecorder()
    statements = [
        chapter_assertion_delete_statement(3),
        NativeCypherBuilder.character_upsert_cypher(CharacterProfile(id="char:01", name="Literal Name"), 3, assertion_origin="chapter_profile"),
        NativeCypherBuilder.world_item_upsert_cypher(WorldItem.from_dict("Location", "Exact Hall", {"id": "location:02"}), 3, assertion_origin="chapter_profile"),
        (database.embedding_update_query, {"identity": {"label": "Character", "id": "char:01", "name": "Literal Name"},
                                           "vector": [0.25, 0.75], "text_hash": "synthetic-hash", "model": "synthetic-model", "embedding_identity": "synthetic-producer"}),
        relationship_statement({"type": "Character", "name": "Literal Name", "id": "char:01"}, "LOCATED_IN",
                               {"type": "Location", "name": "Exact Hall", "id": "location:02"}, 3, origin="chapter_extraction", provisional=False),
        build_chapter_upsert_statement(chapter_number=3),
    ]
    before = deepcopy(statements)
    await database.execute_cypher_batch(statements)
    assert database.batch_statements == [before]
    assert database.executed_queries == before
    assert database.nodes == []
    for query, parameters in statements:
        with pytest.raises(AssertionError, match="Unconfigured synthetic query"):
            await database.execute_cypher_batch([(query + " RETURN 'unknown suffix'", parameters)])
    for query in ("UNWIND $entities AS entity RETURN entity", "CHAR_UPSERT", "CHAPTER_UPSERT", "query"):
        with pytest.raises(AssertionError, match="Unconfigured synthetic query"):
            await database.execute_cypher_batch([(query, {})])
    with pytest.raises(AssertionError, match="Unexpected native write parameters"):
        await database.execute_cypher_batch([(statements[0][0], {"wrong_chapter": 3})])


@pytest.mark.parametrize("factory", [FakeNeo4jManager, StrictCommitRecorder])
async def test_recorders_reject_transactions_without_running_callback(factory: type[FakeNeo4jManager]) -> None:
    database = factory()
    calls = []

    def callback(transaction: Any) -> None:
        calls.append(transaction)

    with pytest.raises(AssertionError, match="Response-only fake cannot execute a transaction"):
        await database.execute_in_transaction(callback)
    assert calls == []
