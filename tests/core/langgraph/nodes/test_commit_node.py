# tests/core/langgraph/nodes/test_commit_node.py
"""Tests for core/langgraph/nodes/commit_node.py - entity and relationship persistence."""

from collections.abc import Callable, Generator
from copy import deepcopy
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from neo4j import Driver
from structlog.testing import capture_logs

import config
from core.db_manager import Neo4jManagerSingleton
from core.graph_ownership import OWNER_QUERY
from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.commit_node import _build_entity_persistence_statements, commit_to_graph
from core.langgraph.state import NarrativeState
from core.service_context import get_services
from data_access import character_queries, kg_queries, world_queries
from data_access.cypher_builders.native_builders import NativeCypherBuilder, chapter_assertion_delete_statement
from models.kg_models import CharacterProfile, WorldItem
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager
from tests.fakes.service_context import configure_empty_entity_names, patch_service

pytestmark = pytest.mark.usefixtures("offline_commit_providers")


@pytest.fixture(autouse=True)
def known_entity_names(offline_commit_providers: FakeNeo4jManager) -> None:
    configure_empty_entity_names(offline_commit_providers)


@pytest.mark.parametrize(
    ("character_names", "location_names", "chapter"),
    [([], [], 0), (["Alice"], [], 0), (["Alice", "Bob"], [], 7),
     ([], ["Castle"], 5), ([], ["Castle", "Forest"], 5), (["Alice"], ["Castle"], 7)],
)
@pytest.mark.run_settings(ENABLE_ENTITY_EMBEDDING_PERSISTENCE=False)
async def test_native_entity_provider_preserves_payloads(
    character_names: list[str], location_names: list[str], chapter: int,
    offline_commit_providers: FakeNeo4jManager,
) -> None:
    characters = [CharacterProfile(name=name, personality_description=f"About {name}", traits=["brave"]) for name in character_names]
    locations = [WorldItem.from_dict("Location", name, {"description": f"About {name}"}) for name in location_names]

    statements = await _build_entity_persistence_statements(characters, locations, chapter)

    assert statements == [
        *[NativeCypherBuilder.character_upsert_cypher(character, chapter, assertion_origin="chapter_profile") for character in characters],
        *[NativeCypherBuilder.world_item_upsert_cypher(location, chapter, assertion_origin="chapter_profile") for location in locations],
    ]
    assert [(parameters["name"], parameters["description"], parameters["chapter_number"]) for _, parameters in statements] == [
        (name, f"About {name}", chapter) for name in character_names + location_names
    ]
    assert [parameters["trait_data"] for _, parameters in statements[:len(characters)]] == [["brave"] for _ in characters]
    assert offline_commit_providers.executed_queries == []
    assert offline_commit_providers.batch_statements == []


@pytest.mark.run_settings(ENABLE_ENTITY_EMBEDDING_PERSISTENCE=True, MAIN_NOVEL_INFO_NODE_ID="synthetic_novel")
async def test_commit_batches_real_conversions_embeddings_and_chapter(
    tmp_path: Path, offline_commit_providers: FakeNeo4jManager,
) -> None:
    content_manager = ContentManager(str(tmp_path))
    entities = {
        "characters": [{"name": "Alice", "type": "Character", "description": "A scout", "first_appearance_chapter": 2,
                        "attributes": {"traits": ["brave"], "relationships": {"Bob": {"type": "FRIEND_OF", "description": "Companion"}}}}],
        "world_items": [{"name": "Castle", "type": "Location", "description": "A fortress", "first_appearance_chapter": 2,
                         "attributes": {"category": "location", "rules": ["Be quiet"]}}],
    }
    state: NarrativeState = {
        "project_dir": str(tmp_path), "current_chapter": 7,
        "extracted_entities_ref": content_manager.save_json(entities, "extracted_entities", "chapter_7", 1),
        "extracted_relationships_ref": content_manager.save_json([], "extracted_relationships", "chapter_7", 1),
        "draft_ref": content_manager.save_text("Alice entered the castle.", "draft", "chapter_7", 1),
    }
    original_state = dict(state)

    result = await commit_to_graph(state)

    assert result == {"current_node": "commit_to_graph", "last_error": None, "has_fatal_error": False}
    assert state == original_state
    assert len(offline_commit_providers.batch_statements) == 1
    statements = offline_commit_providers.batch_statements[0]
    assert len(statements) == 7
    character_parameters = statements[1][1]
    location_parameters = statements[2][1]
    assert (character_parameters["name"], character_parameters["description"], character_parameters["trait_data"],
            character_parameters["created_chapter"], character_parameters["chapter_number"]) == ("Alice", "A scout", ["brave"], 2, 7)
    assert character_parameters["relationship_data"] == []
    assert (statements[5][1]["object_name"], statements[5][1]["predicate_clean"], statements[5][1]["assertion_origin"]) == ("Bob", "FRIEND_OF", "chapter_profile")
    assert statements[5][1]["relationship_properties"]["description"] == "Companion"
    assert (location_parameters["name"], location_parameters["description"], location_parameters["rules"],
            location_parameters["created_chapter"], location_parameters["chapter_number"]) == ("Castle", "A fortress", ["Be quiet"], 2, 7)
    assert [parameters["vector"] for _, parameters in statements[3:5]] == [[0.25, 0.75], [0.25, 0.75]]
    assert statements[3][1]["identity"] == {"label": "Character", "id": None, "name": "Alice"}
    assert statements[4][1]["identity"] == {"label": "Location", "id": location_parameters["id"], "name": "Castle"}
    assert statements[0] == chapter_assertion_delete_statement(7)
    assert statements[6][1] == {
        "chapter_number_param": 7, "chapter_id_param": "chapter_synthetic_novel_7", "summary_param": None,
        "embedding_vector_param": None, "is_provisional_param": False, "title_param": None, "act_number_param": None,
        "embedding_model_param": None, "embedding_identity_param": None,
        "generation_status_param": None,
    }


class ChapterTransaction:
    """Model only the chapter-write subset, not Cypher or Neo4j semantics."""

    def __init__(self, database: "ChapterDriver") -> None:
        self.database = database
        self.pending = deepcopy(database.graph)
        self.events: list[str] = []
        self.finished = False

    def run(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        from tests.fakes.graph_ownership import OwnershipRows
        from tests.fakes.schema_catalog import schema_catalog

        catalog = schema_catalog()
        if query in catalog:
            return catalog[query]
        if query == OWNER_QUERY:
            return [{"key": "exclusive", "project_id": "11111111-1111-4111-8111-111111111111", "version": 1}]
        if query == "MATCH (owner:SagaGraphOwner {key: 'exclusive', project_id: $project_id}) SET owner.version = owner.version":
            assert parameters == {"project_id": "11111111-1111-4111-8111-111111111111"}
            return OwnershipRows([])
        assert parameters is not None
        statement = " ".join(query.split())
        if statement == "MATCH (n) WHERE n:Character OR n:Location OR n:Event OR n:Item RETURN DISTINCT toLower(n.name) AS name":
            return [{"name": name.lower()} for name in self.pending["entities"]]
        self.events.append("run")
        if len(self.events) == 2 and self.database.failure in ("statement", "driver_rollback"):
            if self.database.failure == "driver_rollback":
                self.pending = deepcopy(self.database.graph)
                self.finished = True
                self.events.append("driver_rollback")
            self.database.failure = "none"
            raise RuntimeError("synthetic statement failure")
        if statement == " ".join(chapter_assertion_delete_statement(parameters.get("chapter", 0))[0].split()):
            self.pending["relationships"] = [relationship for relationship in self.pending["relationships"]
                                              if not (relationship["chapter_added"] == parameters["chapter"]
                                                      and relationship.get("assertion_origin") in {"chapter_extraction", "chapter_profile"})]
        elif statement == "MATCH (n) WHERE n.created_chapter = $chapter AND NOT (n)--() DELETE n":
            connected = {relationship[endpoint] for relationship in self.pending["relationships"] for endpoint in ("source", "target")}
            self.pending["entities"] = {name: fields for name, fields in self.pending["entities"].items() if fields["created_chapter"] != parameters["chapter"] or name in connected}
        elif statement == "MATCH (c:Chapter {number: $chapter}) DELETE c":
            self.pending["chapters"].pop(parameters["chapter"], None)
        elif statement.startswith("MERGE (c:Chapter {number: $chapter_number_param})"):
            chapter = parameters["chapter_number_param"]
            fields = self.pending["chapters"].setdefault(chapter, {"number": chapter, "id": parameters["chapter_id_param"]})
            for name in ("summary", "is_provisional"):
                if parameters[f"{name}_param"] is not None:
                    fields[name] = parameters[f"{name}_param"]
        else:
            raise AssertionError(f"Unsupported synthetic statement: {statement}")
        return []

    def commit(self) -> None:
        if self.database.failure == "commit_before":
            self.database.failure = "none"
            raise RuntimeError("synthetic commit failure")
        self.database.graph = deepcopy(self.pending)
        self.events.append("commit")
        self.finished = True
        if self.database.failure == "commit_acknowledgement":
            self.database.failure = "none"
            raise RuntimeError("synthetic acknowledgement failure")

    def rollback(self) -> None:
        assert not self.finished
        self.pending = deepcopy(self.database.graph)
        self.events.append("rollback")
        self.finished = True

    def closed(self) -> bool:
        return self.finished


class ChapterDriver:
    def __init__(self) -> None:
        self.failure = "none"
        self.transactions: list[ChapterTransaction] = []
        self.graph: dict[str, Any] = {
            "chapters": {4: {"number": 4, "id": "prior-chapter-4", "summary": "Prior committed summary", "is_provisional": False}},
            "entities": {
                "Alice": {"description": "Prior committed scout", "created_chapter": 4},
                "Bob": {"description": "Prior committed guide", "created_chapter": 3},
                "Keepsake": {"description": "Prior committed orphan", "created_chapter": 4},
            },
            "relationships": [{"source": "Alice", "target": "Bob", "type": "KNOWS", "chapter_added": 4, "assertion_origin": "chapter_extraction", "description": "Prior committed fact"}],
        }

    def session(self, *, database: str) -> "ChapterDriver":
        return self

    def __enter__(self) -> "ChapterDriver":
        return self

    def __exit__(self, *arguments: Any) -> None:
        pass

    def execute_read(self, operation: Callable[..., Any], *arguments: Any) -> Any:
        return operation(ChapterTransaction(self), *arguments)

    def begin_transaction(self) -> ChapterTransaction:
        transaction = ChapterTransaction(self)
        self.transactions.append(transaction)
        return transaction


@pytest.fixture
def chapter_driver(monkeypatch: pytest.MonkeyPatch) -> Generator[ChapterDriver, None, None]:
    driver = ChapterDriver()
    manager = object.__new__(Neo4jManagerSingleton)
    manager._initialized_flag = False
    Neo4jManagerSingleton.__init__(manager)
    manager.bind_project("11111111-1111-4111-8111-111111111111")
    manager.driver = cast(Driver, driver)
    monkeypatch.setattr(get_services(), 'database', manager)
    assert Path(commit_to_graph.__code__.co_filename).resolve() == Path(__file__).resolve().parents[4] / "core/langgraph/nodes/commit_node.py"
    assert Path(Neo4jManagerSingleton.execute_cypher_batch.__code__.co_filename).resolve() == Path(__file__).resolve().parents[4] / "core/db_manager.py"
    enclosing = config.snapshot_settings()
    values = {name: getattr(enclosing, name) for name in config.EffectiveSettings.model_fields}
    with config.bind_settings(config.EffectiveSettings(_env_file=None, **{**values, "MAIN_NOVEL_INFO_NODE_ID": "synthetic_novel"})):
        yield driver


def chapter_state(directory: Path, chapter: int = 4) -> NarrativeState:
    content_manager = ContentManager(str(directory))
    return {
        "project_dir": str(directory), "current_chapter": chapter,
        "draft_ref": content_manager.save_text("A synthetic draft.", "draft", f"chapter_{chapter}", 1),
    }


@pytest.fixture
def observed_cache_clears(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def observe(group: str, clear: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            calls.append(group)
            clear()
        return run

    for group, reader in (
        ("character", character_queries.get_character_profile_by_name),
        ("world", world_queries.get_world_item_by_id),
        ("kg", kg_queries.query_kg_from_db),
    ):
        monkeypatch.setattr(reader, "cache_clear", observe(group, cast(Any, reader).cache_clear))
    return calls


async def test_preparation_failure_preserves_prior_commit(tmp_path: Path, chapter_driver: ChapterDriver, observed_cache_clears: list[str]) -> None:
    state = chapter_state(tmp_path)
    reference = state["draft_ref"]
    assert reference is not None
    state["draft_ref"] = {**reference, "checksum": "0" * 64}
    before = deepcopy(chapter_driver.graph)
    original_state = deepcopy(state)

    for _ in range(2):
        result = await commit_to_graph(state)

        assert chapter_driver.graph == before
        assert chapter_driver.transactions == []
        assert observed_cache_clears == []
        assert result["has_fatal_error"] is True
        assert result["error_node"] == "commit"
        assert result["current_node"] == "commit_to_graph"
        assert state == original_state


@pytest.mark.parametrize("failure", ["statement", "driver_rollback", "commit_before"])
async def test_commit_reports_batch_failure_with_real_content(tmp_path: Path, chapter_driver: ChapterDriver, failure: str, observed_cache_clears: list[str]) -> None:
    state = chapter_state(tmp_path)
    chapter_driver.failure = failure
    before = deepcopy(chapter_driver.graph)
    expected_events = ["run", "run", "driver_rollback" if failure == "driver_rollback" else "rollback"]

    for attempt in range(2):
        chapter_driver.failure = failure
        result = await commit_to_graph(state)

        assert chapter_driver.graph == before
        assert [transaction.events for transaction in chapter_driver.transactions] == [expected_events] * (attempt + 1)
        assert observed_cache_clears == []
        assert result == {
            "current_node": "commit_to_graph", "has_fatal_error": True, "error_node": "commit",
            "last_error": "Commit to graph failed: Batch Cypher execution failed (Details: "
            + str({"batch_size": 2, "error_code": "UNKNOWN", "error_message": "synthetic commit failure" if failure == "commit_before" else "synthetic statement failure",
                   "original_error": "synthetic commit failure" if failure == "commit_before" else "synthetic statement failure", "operation": "batch_execution"}) + ")",
        }

    chapter_driver.failure = "none"
    expected = deepcopy(before)
    expected["relationships"] = []
    assert await commit_to_graph(state) == {"current_node": "commit_to_graph", "last_error": None, "has_fatal_error": False}
    assert chapter_driver.graph == expected


@pytest.mark.parametrize("chapter", [4, 5])
@pytest.mark.parametrize("cache", ["character_queries.get_character_profile_by_name", "world_queries.get_world_item_by_id", "kg_queries.query_kg_from_db"])
async def test_cache_failure_preserves_durable_commit(
    tmp_path: Path, chapter_driver: ChapterDriver, monkeypatch: pytest.MonkeyPatch, chapter: int, cache: str, observed_cache_clears: list[str],
) -> None:
    state = chapter_state(tmp_path, chapter)
    expected = deepcopy(chapter_driver.graph)
    if chapter == 4:
        expected["relationships"] = []
    else:
        expected["chapters"][5] = {"number": 5, "id": "chapter_synthetic_novel_5", "is_provisional": False}

    group = cache.split("_queries.")[0]

    def fail_cache_clear() -> None:
        observed_cache_clears.append(group)
        raise RuntimeError("synthetic cache failure")

    with monkeypatch.context() as context:
        context.setattr(f"data_access.{cache}.cache_clear", fail_cache_clear)
        with capture_logs() as logs:
            result = await commit_to_graph(state)
        assert chapter_driver.graph == expected
        assert [transaction.events for transaction in chapter_driver.transactions] == [["run", "run", "commit"]]
        assert result == {"current_node": "commit_to_graph", "last_error": None, "has_fatal_error": False}
        warnings = [entry for entry in logs if entry["event"] == "commit_to_graph: postcommit cache invalidation failed"]
        assert observed_cache_clears == ["character", "world", "kg"]
        assert warnings == [{"event": "commit_to_graph: postcommit cache invalidation failed", "chapter": chapter, "cache": group, "error": "synthetic cache failure", "log_level": "warning"}]

    assert await commit_to_graph(state) == {"current_node": "commit_to_graph", "last_error": None, "has_fatal_error": False}
    assert chapter_driver.graph == expected
    assert [transaction.events for transaction in chapter_driver.transactions] == [["run", "run", "commit"], ["run", "run", "commit"]]
    assert observed_cache_clears == ["character", "world", "kg"] * 2


async def test_unavailable_cache_clear_warns_without_failing_commit(
    tmp_path: Path, chapter_driver: ChapterDriver, monkeypatch: pytest.MonkeyPatch, observed_cache_clears: list[str],
) -> None:
    monkeypatch.setattr(character_queries.get_character_profile_by_name, "cache_clear", None)
    with capture_logs() as logs:
        result = await commit_to_graph(chapter_state(tmp_path, 5))
    assert result == {"current_node": "commit_to_graph", "last_error": None, "has_fatal_error": False}
    assert observed_cache_clears == ["world", "kg"]
    assert [transaction.events for transaction in chapter_driver.transactions] == [["run", "run", "commit"]]
    assert [entry for entry in logs if entry["event"] == "commit_to_graph: postcommit cache invalidation failed"] == [{
        "event": "commit_to_graph: postcommit cache invalidation failed", "chapter": 5, "cache": "character",
        "cache_cleared": {"get_character_profile_by_name": False, "get_character_profile_by_id": True}, "log_level": "warning",
    }]


async def test_unknown_commit_acknowledgement_never_compensates(tmp_path: Path, chapter_driver: ChapterDriver) -> None:
    state = chapter_state(tmp_path)
    chapter_driver.failure = "commit_acknowledgement"
    expected = deepcopy(chapter_driver.graph)
    expected["relationships"] = []

    result = await commit_to_graph(state)

    assert chapter_driver.graph == expected
    assert [transaction.events for transaction in chapter_driver.transactions] == [["run", "run", "commit"]]
    assert result["has_fatal_error"] is True
    assert result["error_node"] == "commit"


class TestCommitNodeEntityPersistence:
    """Test entity persistence operations in the commit node."""

    @pytest.mark.asyncio
    async def test_commit_to_graph_creates_entity_ids(self, tmp_path: Path, offline_commit_providers: FakeNeo4jManager) -> None:
        """Submit name-only identities to the native resolver, not a Python ID generator."""
        state = chapter_state(tmp_path, 1)
        manager = ContentManager(str(tmp_path))
        state["extracted_entities_ref"] = manager.save_json({
            "characters": [
                {"name": "Alice", "type": "Character", "description": "Protagonist", "first_appearance_chapter": 1},
                {"name": "Bob", "type": "Character", "description": "Antagonist", "first_appearance_chapter": 1},
            ],
            "world_items": [{"name": "Sword", "type": "Item", "description": "A sharp sword", "first_appearance_chapter": 1}],
        }, "extracted_entities", "chapter_1", 1)

        with patch("utils.text_processing.generate_entity_id") as generate_id:
            result = await commit_to_graph(state)

        assert result == {"current_node": "commit_to_graph", "has_fatal_error": False, "last_error": None}
        assert len(offline_commit_providers.batch_statements) == 1
        statements = offline_commit_providers.batch_statements[0]
        assert [(parameters["name"], parameters["id"]) for _, parameters in statements[1:4]] == [("Alice", None), ("Bob", None), ("Sword", None)]
        assert [parameters["identity"] for _, parameters in statements[4:7]] == [
            {"label": "Character", "id": None, "name": "Alice"},
            {"label": "Character", "id": None, "name": "Bob"},
            {"label": "Item", "id": None, "name": "Sword"},
        ]
        generate_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_commit_to_graph_handles_empty_extractions(self) -> None:
        """Test that commit_to_graph handles empty extraction results."""
        mock_state = {
            "current_chapter": 1,
            "project_dir": "/tmp/test_project",
            "extracted_entities_ref": {
                "path": ".saga/content/extracted_entities/chapter_1.json",
                "content_type": "extracted_entities",
                "version": 1,
                "size_bytes": 0,
                "checksum": "empty",
            },
        }

        with patch("core.langgraph.nodes.commit_node.ContentManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm_class.return_value = mock_cm
            mock_cm.load_json_strict.return_value = {
                "characters": [],
                "world_items": [],
            }
            mock_cm.load_text_strict.return_value = "Test draft text"

            with patch_service('database.execute_cypher_batch') as mock_execute:
                # Add draft_ref to state so get_draft_text doesn't fail
                mock_state["draft_ref"] = {
                    "path": ".saga/content/drafts/chapter_1.txt",
                    "content_type": "draft",
                    "version": 1,
                    "size_bytes": 10,
                    "checksum": "draft123",
                }
                await commit_to_graph(mock_state)  # type: ignore[arg-type]

                # Should still execute the batch (for chapter node creation)
                assert mock_execute.called

    @pytest.mark.asyncio
    async def test_commit_to_graph_deduplicates_entities(self) -> None:
        """Test that commit_to_graph performs entity deduplication."""
        mock_state = {
            "current_chapter": 1,
            "project_dir": "/tmp/test_project",
            "extracted_entities_ref": {
                "path": ".saga/content/extracted_entities/chapter_1.json",
                "content_type": "extracted_entities",
                "version": 1,
                "size_bytes": 100,
                "checksum": "abc123",
            },
        }

        with patch("core.langgraph.nodes.commit_node.ContentManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm_class.return_value = mock_cm
            mock_cm.load_json_strict.return_value = {
                "characters": [
                    {"name": "Alice", "type": "Character", "description": "A scout", "first_appearance_chapter": 1},
                    {"name": "Alice", "type": "Character", "description": "A scout", "first_appearance_chapter": 1},  # Duplicate
                ],
                "world_items": [],
            }

            with patch_service('database.execute_cypher_batch'):
                await commit_to_graph(mock_state)  # type: ignore[arg-type]

                # Should not call generate_entity_id since there are no world items
                # The deduplication logic removes duplicate characters within the batch


class TestCommitNodeRelationshipPersistence:
    """Test relationship persistence operations in the commit node."""

    @pytest.mark.asyncio
    async def test_commit_to_graph_creates_relationships(self) -> None:
        """Test that commit_to_graph creates relationships between entities."""
        mock_state = {
            "current_chapter": 1,
            "project_dir": "/tmp/test_project",
            "extracted_entities_ref": {
                "path": ".saga/content/extracted_entities/chapter_1.json",
                "content_type": "extracted_entities",
                "version": 1,
                "size_bytes": 100,
                "checksum": "abc123",
            },
            "extracted_relationships_ref": {
                "path": ".saga/content/extracted_relationships/chapter_1.json",
                "content_type": "extracted_relationships",
                "version": 1,
                "size_bytes": 50,
                "checksum": "def456",
            },
        }

        with patch("core.langgraph.nodes.commit_node.ContentManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm_class.return_value = mock_cm
            mock_cm.load_json_strict.side_effect = [
                {
                    "characters": [{"name": "Alice", "type": "Character", "description": "A scout", "first_appearance_chapter": 1}],
                    "world_items": [],
                },
                [
                    {
                        "source_name": "Alice",
                        "target_name": "Bob",
                        "relationship_type": "KNOWS",
                        "description": "Acquaintance",
                        "chapter": 1,
                        "confidence": 0.9,
                    },
                ],
            ]
            mock_cm.load_text_strict.return_value = "Test draft text"

            # Add draft_ref to state so get_draft_text doesn't fail
            mock_state["draft_ref"] = {
                "path": ".saga/content/drafts/chapter_1.txt",
                "content_type": "draft",
                "version": 1,
                "size_bytes": 10,
                "checksum": "draft123",
            }

            with patch_service('database.execute_cypher_batch') as mock_execute:
                await commit_to_graph(mock_state)  # type: ignore[arg-type]

                # Verify the batch was executed
                assert mock_execute.called

    @pytest.mark.asyncio
    async def test_commit_to_graph_handles_missing_relationships(self) -> None:
        """Test that commit_to_graph handles missing relationship data."""
        mock_state = {
            "current_chapter": 1,
            "project_dir": "/tmp/test_project",
            "extracted_entities_ref": {
                "path": ".saga/content/extracted_entities/chapter_1.json",
                "content_type": "extracted_entities",
                "version": 1,
                "size_bytes": 100,
                "checksum": "abc123",
            },
            # No extracted_relationships_ref
        }

        with patch("core.langgraph.nodes.commit_node.ContentManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm_class.return_value = mock_cm
            mock_cm.load_json_strict.return_value = {
                "characters": [{"name": "Alice", "type": "Character", "description": "A scout", "first_appearance_chapter": 1}],
                "world_items": [],
            }
            mock_cm.load_text_strict.return_value = "Test draft text"

            # Add draft_ref to state so get_draft_text doesn't fail
            mock_state["draft_ref"] = {
                "path": ".saga/content/drafts/chapter_1.txt",
                "content_type": "draft",
                "version": 1,
                "size_bytes": 10,
                "checksum": "draft123",
            }

            with patch_service('database.execute_cypher_batch') as mock_execute:
                await commit_to_graph(mock_state)  # type: ignore[arg-type]

                # Should still execute the batch (for entities and chapter)
                assert mock_execute.called


class TestCommitNodeChapterPersistence:
    """Test chapter node creation in the commit node."""

    @pytest.mark.asyncio
    async def test_commit_to_graph_creates_chapter_node(self) -> None:
        """Test that commit_to_graph creates a chapter node."""
        mock_state = {
            "current_chapter": 1,
            "project_dir": "/tmp/test_project",
            "extracted_entities_ref": {
                "path": ".saga/content/extracted_entities/chapter_1.json",
                "content_type": "extracted_entities",
                "version": 1,
                "size_bytes": 100,
                "checksum": "abc123",
            },
        }

        with patch("core.langgraph.nodes.commit_node.ContentManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm_class.return_value = mock_cm
            mock_cm.load_json_strict.return_value = {
                "characters": [],
                "world_items": [],
            }
            mock_cm.load_text_strict.return_value = "Test draft text"

            # Add draft_ref to state so get_draft_text doesn't fail
            mock_state["draft_ref"] = {
                "path": ".saga/content/drafts/chapter_1.txt",
                "content_type": "draft",
                "version": 1,
                "size_bytes": 10,
                "checksum": "draft123",
            }

            with patch_service('database.execute_cypher_batch') as mock_execute:
                await commit_to_graph(mock_state)  # type: ignore[arg-type]

                # Verify the batch was executed (should contain chapter creation)
                assert mock_execute.called

    @pytest.mark.asyncio
    async def test_commit_to_graph_links_chapter_to_entities(self) -> None:
        """Test that commit_to_graph links the chapter to created entities."""
        mock_state = {
            "current_chapter": 1,
            "project_dir": "/tmp/test_project",
            "extracted_entities_ref": {
                "path": ".saga/content/extracted_entities/chapter_1.json",
                "content_type": "extracted_entities",
                "version": 1,
                "size_bytes": 100,
                "checksum": "abc123",
            },
        }

        with patch("core.langgraph.nodes.commit_node.ContentManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm_class.return_value = mock_cm
            mock_cm.load_json_strict.return_value = {
                "characters": [{"name": "Alice", "type": "Character", "description": "A scout", "first_appearance_chapter": 1}],
                "world_items": [],
            }
            mock_cm.load_text_strict.return_value = "Test draft text"

            # Add draft_ref to state so get_draft_text doesn't fail
            mock_state["draft_ref"] = {
                "path": ".saga/content/drafts/chapter_1.txt",
                "content_type": "draft",
                "version": 1,
                "size_bytes": 10,
                "checksum": "draft123",
            }

            with patch_service('database.execute_cypher_batch') as mock_execute:
                await commit_to_graph(mock_state)  # type: ignore[arg-type]

                # Verify the batch was executed
                assert mock_execute.called


class TestCommitNodeErrorHandling:
    """Test error handling in the commit node."""

    @pytest.mark.asyncio
    async def test_commit_to_graph_handles_database_errors(self) -> None:
        """Test that commit_to_graph handles database errors gracefully."""
        mock_state = {
            "current_chapter": 1,
            "project_dir": "/tmp/test_project",
            "extracted_entities_ref": {
                "path": ".saga/content/extracted_entities/chapter_1.json",
                "content_type": "extracted_entities",
                "version": 1,
                "size_bytes": 100,
                "checksum": "abc123",
            },
        }

        with patch("core.langgraph.nodes.commit_node.ContentManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm_class.return_value = mock_cm
            mock_cm.load_json_strict.return_value = {
                "characters": [{"name": "Alice", "type": "Character", "description": "A scout", "first_appearance_chapter": 1}],
                "world_items": [],
            }

            with patch_service('database.execute_cypher_batch') as mock_execute:
                mock_execute.side_effect = Exception("Database error")

                result = await commit_to_graph(mock_state)  # type: ignore[arg-type]

                # Should return state with error information
                assert "has_fatal_error" in result
                assert result["has_fatal_error"] is True

    @pytest.mark.asyncio
    async def test_commit_to_graph_handles_missing_content(self) -> None:
        """Test that commit_to_graph handles missing content references."""
        mock_state = {
            "current_chapter": 1,
            "project_dir": "/tmp/test_project",
            # No extracted_entities_ref
        }

        result = await commit_to_graph(mock_state)  # type: ignore[arg-type]

        # Should return state with error information
        assert "has_fatal_error" in result
        assert result["has_fatal_error"] is True


class TestCommitNodeStateManagement:
    """Test state management in the commit node."""

    @pytest.mark.asyncio
    async def test_commit_to_graph_updates_state(self) -> None:
        """Test that commit_to_graph updates the state correctly."""
        mock_state = {
            "current_chapter": 1,
            "project_dir": "/tmp/test_project",
            "extracted_entities_ref": {
                "path": ".saga/content/extracted_entities/chapter_1.json",
                "content_type": "extracted_entities",
                "version": 1,
                "size_bytes": 100,
                "checksum": "abc123",
            },
        }

        with patch("core.langgraph.nodes.commit_node.ContentManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm_class.return_value = mock_cm
            mock_cm.load_json_strict.return_value = {
                "characters": [{"name": "Alice", "type": "Character", "description": "A scout", "first_appearance_chapter": 1}],
                "world_items": [],
            }

            with patch_service('database.execute_cypher_batch'):
                result = await commit_to_graph(mock_state)  # type: ignore[arg-type]

                # Should update current_node
                assert result["current_node"] == "commit_to_graph"

    @pytest.mark.asyncio
    async def test_commit_to_graph_preserves_existing_state(self) -> None:
        """Test that commit_to_graph preserves existing state fields."""
        mock_state = {
            "current_chapter": 1,
            "project_dir": "/tmp/test_project",
            "some_existing_field": "preserve_this",
            "extracted_entities_ref": {
                "path": ".saga/content/extracted_entities/chapter_1.json",
                "content_type": "extracted_entities",
                "version": 1,
                "size_bytes": 100,
                "checksum": "abc123",
            },
        }

        with patch("core.langgraph.nodes.commit_node.ContentManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm_class.return_value = mock_cm
            mock_cm.load_json_strict.return_value = {
                "characters": [],
                "world_items": [],
            }
            mock_cm.load_text_strict.return_value = "Test draft text"

            # Add draft_ref to state so get_draft_text doesn't fail
            mock_state["draft_ref"] = {
                "path": ".saga/content/drafts/chapter_1.txt",
                "content_type": "draft",
                "version": 1,
                "size_bytes": 10,
                "checksum": "draft123",
            }

            with patch_service('database.execute_cypher_batch'):
                result = await commit_to_graph(mock_state)  # type: ignore[arg-type]

                # Should return success state with expected fields
                assert result["current_node"] == "commit_to_graph"
                assert result["has_fatal_error"] is False
