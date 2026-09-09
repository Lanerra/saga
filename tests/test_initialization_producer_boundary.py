"""Producer-boundary regressions for staged initialization."""
import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.langgraph.content_manager import ContentManager
from core.langgraph.initialization.commit_init_node import commit_initialization_to_graph
from core.langgraph.initialization.outline_relationships_node import extract_outline_relationships
from core.langgraph.initialization.staged_import import InitializationImport
from core.langgraph.state import NarrativeState
from core.service_context import get_services
from tests.test_staged_initialization import example_state
from utils.file_io import write_yaml_file


async def test_generated_character_evidence_survives_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = example_state(tmp_path)
    manager = ContentManager(str(tmp_path))
    assert state["character_sheets_ref"] is not None
    sheets = manager.load_json_strict(state["character_sheets_ref"])
    sheets["Ada"].update(generated_at="initialization", raw_response="synthetic exact producer bytes")
    state["character_sheets_ref"] = manager.save_json(sheets, "character_sheets", "all", version=2)
    monkeypatch.setattr(get_services().language_model, "async_call_llm", AsyncMock(return_value=("[]", {})))
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    result = await commit_initialization_to_graph(state)
    assert result["initialization_step"] == "initialization_prepared"
    assert InitializationImport(str(tmp_path)).load().snapshot.source("character_sheets") == sheets


async def test_valid_empty_relationship_extraction_publishes_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = example_state(tmp_path)
    monkeypatch.setattr(get_services().language_model, "async_call_llm", AsyncMock(return_value=('{"kg_triples": []}', {})))
    result = await extract_outline_relationships(state)
    assert isinstance(result["outline_relationships_ref"], dict)
    assert ContentManager(str(tmp_path)).load_json_strict(result["outline_relationships_ref"]) == []


async def test_failed_relationship_extraction_is_not_valid_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = example_state(tmp_path)
    monkeypatch.setattr(get_services().language_model, "async_call_llm", AsyncMock(return_value=('{"kg_triples": [1]}', {})))
    with pytest.raises(ValueError):
        await extract_outline_relationships(state)


async def test_corrupt_projection_manifest_fails_before_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from core.langgraph.initialization import persist_files_node

    state = example_state(tmp_path)
    monkeypatch.setattr(get_services().language_model, "async_call_llm", AsyncMock(return_value=("[]", {})))
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    importer = InitializationImport(str(tmp_path))
    plan = await importer.prepare(state)
    original = persist_files_node._publish_new_file

    def interrupted(path: Path, data: Any, *, writer: Callable[[Path, Any], None] = write_yaml_file) -> None:
        if not path.is_relative_to(tmp_path / ".saga"):
            raise OSError("synthetic before projection")
        original(path, data, writer=writer)

    monkeypatch.setattr(persist_files_node, "_publish_new_file", interrupted)
    with pytest.raises(OSError, match="synthetic before projection"):
        importer.publish_projections(plan)
    monkeypatch.setattr(persist_files_node, "_publish_new_file", original)
    target = tmp_path / f".saga/initialization/{plan.identity}-projections.json"
    manifest = json.loads(target.read_text())
    manifest["world/rules.yaml"] = "rules: [unaccepted injected bytes]\n"
    target.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Projection checksum mismatch"):
        importer.publish_projections(plan)
    assert not (tmp_path / "world/rules.yaml").exists()


@pytest.mark.parametrize("case", ["global_field_type", "global_unknown_field", "producer_validation_errors"])
async def test_global_semantic_schema_precedes_extraction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    state = example_state(tmp_path)
    manager = ContentManager(str(tmp_path))
    assert state["global_outline_ref"] is not None
    value = manager.load_json_strict(state["global_outline_ref"])
    if case == "global_field_type":
        value["thematic_progression"] = 42
    elif case == "global_unknown_field":
        value["unadmitted_semantics"] = "cannot silently discard"
    else:
        value["validation_errors"] = ["producer reports incomplete topology"]
    state["global_outline_ref"] = manager.save_json(value, "global_outline", "all", version=2)
    provider = AsyncMock(return_value=("[]", {}))
    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider)
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    result = await commit_initialization_to_graph(state)
    assert result["has_fatal_error"] is True
    provider.assert_not_awaited()


async def test_selected_user_edits_are_frozen_and_projected_without_reextraction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    import yaml

    state = example_state(tmp_path)
    manager = ContentManager(str(tmp_path))
    assert state["character_sheets_ref"] is not None
    sheets = manager.load_json_strict(state["character_sheets_ref"])
    sheets["Ada"]["motivations"] = "User deliberately selects a different motivation"
    state["character_sheets_ref"] = manager.save_json(sheets, "character_sheets", "all", version=2)
    provider = AsyncMock(return_value=("[]", {}))
    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider)
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    importer = InitializationImport(str(tmp_path))
    plan = await importer.prepare(state)
    assert plan.snapshot.characters[0].motivations == sheets["Ada"]["motivations"]
    projections = json.loads(plan.projections)
    character = yaml.safe_load(projections["characters/ada.yaml"])
    assert character["motivations"] == sheets["Ada"]["motivations"]
    calls = provider.await_count
    importer.publish_projections(plan)
    assert (tmp_path / "characters/ada.yaml").read_text() == projections["characters/ada.yaml"]
    assert await importer.prepare({**state, "initialization_id": plan.identity}) == plan
    assert provider.await_count == calls


async def test_frozen_character_identity_reaches_native_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from utils.text_processing import generate_entity_id

    monkeypatch.setattr(get_services().language_model, "async_call_llm", AsyncMock(return_value=("[]", {})))
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    plan = await InitializationImport(str(tmp_path)).prepare(example_state(tmp_path))
    character = next(entity for entity in plan.entities if entity.label == "Character")
    payload = json.loads(character.payload)
    parameters = [json.loads(statement.parameters) for statement in plan.statements]
    writer = next(value for value in parameters if value.get("name") == "Ada" and "trait_data" in value)
    assert character.identity == generate_entity_id("Ada", "character")
    assert writer["id"] == character.identity
    assert payload["id"] == character.identity


async def test_initial_chapter_rows_satisfy_lifecycle_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    monkeypatch.setattr(get_services().language_model, "async_call_llm", AsyncMock(return_value=("[]", {})))
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    plan = await InitializationImport(str(tmp_path)).prepare(example_state(tmp_path))
    statements = [json.loads(statement.parameters)["properties"] for statement in plan.statements if statement.query.startswith("CREATE (n:Chapter)")]
    assert len(statements) == 1
    assert statements[0]["generation_status"] == "planned"


async def test_files_alone_do_not_admit_orchestrator_initialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.langgraph.initialization.persist_files_node import persist_initialization_files
    from data_access.chapter_queries import ChapterProgress
    from orchestration.langgraph_orchestrator import LangGraphOrchestrator

    state = example_state(tmp_path)
    result = await persist_initialization_files(state)
    assert result["initialization_step"] == "files_persisted"
    monkeypatch.setattr("orchestration.langgraph_orchestrator.chapter_queries.load_chapter_progress_from_db", AsyncMock(return_value=ChapterProgress(0, ())))
    orchestrator = LangGraphOrchestrator(project_dir=tmp_path)
    with pytest.raises(ValueError, match="initialization receipt"):
        await orchestrator._load_or_create_state(project_id="synthetic", narrative_config=None)


def example_relationship_state(tmp_path: Path, channels: str, confidences: tuple[float, ...]) -> NarrativeState:
    state = example_state(tmp_path)
    manager = ContentManager(str(tmp_path))
    assert state["character_sheets_ref"] is not None
    sheets = manager.load_json_strict(state["character_sheets_ref"])
    sheets["Bea"] = copy.deepcopy(sheets["Ada"])
    sheets["Bea"].update(name="Bea", is_protagonist=False)
    if channels in {"profile", "both"}:
        sheets["Ada"]["relationships"] = {"Bea": {"type": "FRIEND_OF", "description": "Trusted companions"}}
    state["character_sheets_ref"] = manager.save_json(sheets, "character_sheets", "all", version=2)
    if channels in {"outline", "both"}:
        state["outline_relationships_ref"] = manager.save_json([
            {"source_name": "Ada", "target_name": "Bea", "relationship_type": "FRIEND_OF", "description": "Trusted companions", "chapter": 0, "confidence": confidence}
            for confidence in confidences
        ], "outline_relationships", "all", version=2)
    return state


@pytest.mark.parametrize("role", ["protagonist", None])
async def test_prompt_nullable_event_role_prepares(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str | None) -> None:
    async def provider(**arguments: Any) -> tuple[str, dict[str, Any]]:
        if arguments["prompt"].startswith("Extract character names"):
            return json.dumps([{"name": "Ada", "role": role}]), {}
        return "[]", {}

    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider)
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    result = await commit_initialization_to_graph(example_state(tmp_path))
    assert result["initialization_step"] == "initialization_prepared", result
    plan = InitializationImport(str(tmp_path)).load()
    act_events = {entity.identity for entity in plan.entities if entity.label == "Event" and json.loads(entity.payload)["event_type"] == "ActKeyEvent"}
    involved = [json.loads(statement.parameters) for statement in plan.statements if "relationship:INVOLVES" in statement.query]
    act_involvements = [edge for edge in involved if edge["source"] in act_events]
    assert len(act_involvements) == 5
    properties: dict[str, Any] = {"chapter_added": 0, "is_provisional": False}
    if role is not None:
        properties["role"] = role
    assert [edge["properties"] for edge in act_involvements] == [properties] * 5


@pytest.mark.parametrize(("channels", "confidences"), [
    ("profile", ()), ("outline", (0.8,)), ("both", (0.8,)),
    ("outline", (0.8, 0.8)), ("both", (0.6, 0.9, 0.6)), ("both", (0.9, 0.6, 0.6)),
])
async def test_compatible_relationship_channels_coalesce(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, channels: str, confidences: tuple[float, ...]) -> None:
    state = example_relationship_state(tmp_path, channels, confidences)
    provider = AsyncMock(return_value=("[]", {}))
    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider)
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    result = await commit_initialization_to_graph(state)
    assert result["initialization_step"] == "initialization_prepared", result
    importer = InitializationImport(str(tmp_path))
    plan = importer.load()
    edges = [json.loads(statement.parameters) for statement in plan.statements if "relationship:FRIEND_OF" in statement.query]
    properties: dict[str, Any] = {
        "description": "Trusted companions", "chapter_added": 0, "is_provisional": False,
        "description_status": "single_description",
        "description_assertions": [
            json.dumps({"source": source, "description": "Trusted companions"}, sort_keys=True, separators=(",", ":"))
            for source in ("profile", "outline") if channels in {source, "both"}
        ],
    }
    if channels in {"profile", "both"}:
        properties.update(type="FRIEND_OF", source_profile_managed=True)
    if channels in {"outline", "both"}:
        properties["confidence"] = max(confidences)
    identities = {json.loads(entity.payload)["name"]: entity.identity for entity in plan.entities if entity.label == "Character"}
    assert edges == [{"source": identities["Ada"], "target": identities["Bea"], "properties": properties}]
    calls = provider.await_count
    assert await importer.prepare({**state, "initialization_id": plan.identity}) == plan
    assert provider.await_count == calls


@pytest.mark.parametrize("case", ["invalid_description", "unknown_endpoint", "unknown_type", "extra_authority"])
async def test_incompatible_relationship_channels_reject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    state = example_relationship_state(tmp_path, "both", (0.8,))
    manager = ContentManager(str(tmp_path))
    assert state["outline_relationships_ref"] is not None
    relationships = manager.load_json_strict(state["outline_relationships_ref"])
    if case == "invalid_description":
        relationships[0]["description"] = 3
    elif case == "unknown_endpoint":
        relationships[0]["target_name"] = "Unknown"
    elif case == "unknown_type":
        relationships[0]["relationship_type"] = "NOT_A_RELATIONSHIP"
    else:
        relationships[0]["source_profile_managed"] = True
    state["outline_relationships_ref"] = manager.save_json(relationships, "outline_relationships", "all", version=3)
    monkeypatch.setattr(get_services().language_model, "async_call_llm", AsyncMock(return_value=("[]", {})))
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    result = await commit_initialization_to_graph(state)
    assert result["initialization_step"] == "commit_failed"
    assert result["has_fatal_error"] is True
    assert not (tmp_path / ".saga/initialization/selected").exists()


@pytest.mark.parametrize(("profile_description", "outline_description"), [
    (
        "Mara's husband, who supports her quietly but sometimes worries her fixation on the ledger keeps her from truly joining the community.",
        "Mara is Tom's wife; they married into their shared life in the valley.",
    ),
    ("Trusted companions", "Bitter enemies"),
    ("  Exact whitespace\n", "Exact whitespace"),
])
async def test_differing_relationship_prose_is_attributed_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile_description: str, outline_description: str,
) -> None:
    from core.db_manager import Neo4jManagerSingleton
    from core.langgraph.initialization.graph_plan import InitializationGraphPlan, produce_plan

    state = example_relationship_state(tmp_path, "both", (0.6, 0.9, 0.6))
    manager = ContentManager(str(tmp_path))
    assert state["character_sheets_ref"] is not None
    sheets = manager.load_json_strict(state["character_sheets_ref"])
    sheets["Ada"]["relationships"]["Bea"] = {"type": "FAMILY_OF", "description": profile_description}
    state["character_sheets_ref"] = manager.save_json(sheets, "character_sheets", "all", version=3)
    assert state["outline_relationships_ref"] is not None
    relationships = manager.load_json_strict(state["outline_relationships_ref"])
    for relationship in relationships:
        relationship.update(relationship_type="FAMILY_OF", description=outline_description)
    state["outline_relationships_ref"] = manager.save_json(relationships, "outline_relationships", "all", version=3)
    provider = AsyncMock(return_value=("[]", {}))
    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider)
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    importer = InitializationImport(str(tmp_path))
    plan = await importer.prepare(state)
    edges = [json.loads(statement.parameters) for statement in plan.statements if "relationship:FAMILY_OF" in statement.query]
    assert len(edges) == 1
    assert edges[0]["properties"] == {
        "chapter_added": 0, "is_provisional": False, "type": "FAMILY_OF", "source_profile_managed": True,
        "confidence": 0.9, "description": profile_description, "description_status": "unresolved",
        "description_assertions": [
            json.dumps({"source": source, "description": description}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            for source, description in [("profile", profile_description), ("outline", outline_description)]
        ],
    }
    assert InitializationGraphPlan.model_validate_json(plan.model_dump_json()) == plan
    assert importer.load() == plan
    calls = provider.await_count
    assert await importer.prepare({**state, "initialization_id": plan.identity}) == plan
    assert provider.await_count == calls
    forward = await produce_plan(plan.snapshot)
    reverse = await produce_plan(plan.snapshot.model_copy(update={"relationships": tuple(reversed(plan.snapshot.relationships))}))
    assert forward.statements == reverse.statements
    with pytest.raises(ValueError, match="Initialization acceptance identity mismatch"):
        await importer.accept("0" * 64)
    written: list[tuple[str, dict[str, Any]]] = []

    def run_statement(query: str, parameters: dict[str, Any] | None = None, **keywords: Any) -> MagicMock:
        result = MagicMock()
        if parameters is not None:
            written.append((query, parameters))
        elif "RETURN owner.initialization_plan AS identity" in query:
            result.single.return_value = {"identity": None}
        elif "WHERE NOT n:SagaGraphOwner" in query:
            result.single.return_value = {"count": 0}
        elif "RETURN count(n) AS count" in query:
            assert keywords["identity"] in {entity.identity for entity in plan.entities}
            result.single.return_value = {"count": 1}
        else:
            assert "SET owner.initialization_plan = $identity" in query
            assert keywords == {"project_id": plan.snapshot.project_id, "identity": plan.identity}
        return result

    async def recording_transport(callback: Callable[[Any], None]) -> None:
        callback(MagicMock(run=run_statement))

    monkeypatch.setattr(Neo4jManagerSingleton, "_instance", None)
    database = Neo4jManagerSingleton()
    monkeypatch.setattr(get_services(), "database", database)
    monkeypatch.setattr(database, "execute_in_transaction", recording_transport)
    monkeypatch.setattr(database, "execute_read_query", AsyncMock(side_effect=[[{"identity": None}], [{"identity": plan.identity}], [{"identity": plan.identity}]]))
    assert await importer.accept(plan.identity) == plan
    assert written == [(statement.query, json.loads(statement.parameters)) for statement in plan.statements]
    assert await importer.accept(plan.identity) == plan
    assert len(written) == len(plan.statements)
    assert provider.await_count == calls + len(forward.evidence) + len(reverse.evidence)
    frozen = tmp_path / f".saga/initialization/{plan.identity}.json"
    frozen.write_text(frozen.read_text().replace("unresolved", "coherent"))
    with pytest.raises(ValueError, match="Frozen initialization checksum mismatch"):
        importer.load()


async def test_outline_only_description_order_does_not_choose_display(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.langgraph.initialization.graph_plan import produce_plan
    from core.langgraph.initialization.snapshot import select_snapshot

    state = example_relationship_state(tmp_path, "outline", (0.9, 0.6, 0.8))
    manager = ContentManager(str(tmp_path))
    assert state["outline_relationships_ref"] is not None
    relationships = manager.load_json_strict(state["outline_relationships_ref"])
    for relationship, description in zip(relationships, ["Zulu", "Alpha", "Zulu"], strict=True):
        relationship["description"] = description
    state["outline_relationships_ref"] = manager.save_json(relationships, "outline_relationships", "all", version=3)
    monkeypatch.setattr(get_services().language_model, "async_call_llm", AsyncMock(return_value=("[]", {})))
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    snapshot = select_snapshot(state)
    forward = await produce_plan(snapshot)
    reverse = await produce_plan(snapshot.model_copy(update={"relationships": tuple(reversed(snapshot.relationships))}))
    assert forward.statements == reverse.statements
    properties = next(json.loads(statement.parameters)["properties"] for statement in forward.statements if "relationship:FRIEND_OF" in statement.query)
    assert properties["description"] == "Alpha"
    assert properties["description_status"] == "unresolved"
    assert properties["confidence"] == 0.9
    assert [json.loads(assertion) for assertion in properties["description_assertions"]] == [
        {"source": "outline", "description": "Alpha"}, {"source": "outline", "description": "Zulu"},
    ]


async def test_structured_relationship_conflict_still_rejects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def provider(**arguments: Any) -> tuple[str, dict[str, Any]]:
        if arguments["prompt"].startswith("Extract character names"):
            return json.dumps([{"name": "Ada", "role": "protagonist"}, {"name": "Ada", "role": "participant"}]), {}
        return "[]", {}

    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider)
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    with pytest.raises(ValueError, match="Conflicting duplicate relationship"):
        await InitializationImport(str(tmp_path)).prepare(example_state(tmp_path))
    assert not (tmp_path / ".saga/initialization/selected").exists()


@pytest.mark.parametrize("involvement", [{"name": "Unknown", "role": None}, {"name": "Ada", "role": 3}, {"name": "Ada"}, {"name": "Ada", "role": None, "extra": True}])
async def test_nullable_role_retains_identity_and_shape_admission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, involvement: dict[str, Any]) -> None:
    async def provider(**arguments: Any) -> tuple[str, dict[str, Any]]:
        if arguments["prompt"].startswith("Extract character names"):
            return json.dumps([involvement]), {}
        return "[]", {}

    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider)
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    result = await commit_initialization_to_graph(example_state(tmp_path))
    assert result["initialization_step"] == "commit_failed"
    assert result["has_fatal_error"] is True
    assert not (tmp_path / ".saga/initialization/selected").exists()
