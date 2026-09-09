"""Synthetic catalog-ID responses test mechanics, not captured/model identity quality."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core.exceptions import ContentIntegrityError
from core.langgraph.content_manager import ContentManager
from core.langgraph.initialization.catalog import materialize_initialization_catalog, select_catalog
from core.langgraph.initialization.outline_relationships_node import extract_outline_relationships
from core.langgraph.initialization.staged_import import InitializationImport
from core.schema_readiness import CONSTRAINT_QUERY
from core.service_context import get_services
from tests.fakes.schema_catalog import schema_catalog
from tests.test_staged_initialization import example_state


async def test_relationship_producer_requires_selected_catalog_before_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = example_state(tmp_path)
    provider = AsyncMock(return_value=('{"kg_triples": []}', {}))
    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider)
    with pytest.raises(ValueError, match="selected initialization_catalog_ref"):
        await extract_outline_relationships(state)
    provider.assert_not_awaited()


def test_import_provenance() -> None:
    import core.langgraph.initialization.graph_plan as module

    assert Path(module.__file__).resolve() == Path(__file__).resolve().parents[1] / "core/langgraph/initialization/graph_plan.py"


class SyntheticSelector:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.assertions: list[dict[str, Any]] = []
        self.responses: dict[str, Any] = {}

    async def __call__(self, **arguments: Any) -> tuple[str, dict[str, Any]]:
        prompt = arguments["prompt"]
        self.prompts.append(prompt)
        for prefix, response in self.responses.items():
            if prompt.startswith(prefix):
                return json.dumps(response), {}
        if "SELECTED ENTITY CATALOG" in prompt:
            return json.dumps({"kg_triples": self.assertions}), {}
        if prompt.startswith("Catalog possessions"):
            return '{"possessions": []}', {}
        if prompt.startswith("Catalog event participants"):
            return "[]", {}
        if prompt.startswith("Catalog event location"):
            return '{"location_id": null}', {}
        if prompt.startswith("Catalog event items"):
            return '{"featured_items": []}', {}
        return json.dumps([
            {"name": "Weather Station", "category": "location", "description": "The valley measuring station."},
            {"name": "Brass Rain Gauge", "category": "object", "description": "The measuring instrument."},
        ]), {}


async def test_catalog_producer_snapshot_plan_reuses_exact_entities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = example_state(tmp_path)
    state["outline_relationships_ref"] = None
    provider = SyntheticSelector()
    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider)
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    state.update(await materialize_initialization_catalog(state))
    catalog = select_catalog(state)
    location = catalog.candidates("Location")[0]
    event = catalog.candidates("Event")[0]
    provider.assertions = [{"source_id": event["id"], "source_label": "Event", "target_id": location["id"], "target_label": "Location", "relationship_type": "OCCURS_AT", "description": "The departure occurs at the station."}]
    state.update(await extract_outline_relationships(state))
    importer = InitializationImport(str(tmp_path))
    plan = await importer.prepare(state)
    assert plan.snapshot.schema_version == 2
    assert {entity.identity: entity for entity in plan.entities} == {entity.identity: entity for entity in catalog.entities}
    assert sum("world_items_lines" in evidence.template for evidence in plan.evidence) == 1
    assert plan.snapshot.relationships[0].model_dump()["source_id"] == event["id"]
    count = len(provider.prompts)
    assert await InitializationImport(str(tmp_path)).prepare(importer.state(plan)) == plan
    assert len(provider.prompts) == count
    reference = state["initialization_catalog_ref"]
    assert reference is not None
    assert ContentManager(str(tmp_path)).load_json_strict(reference) == catalog.model_dump(mode="json")


async def selected_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: SyntheticSelector) -> dict[str, Any]:
    state = example_state(tmp_path)
    state["outline_relationships_ref"] = None
    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider)
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    state.update(await materialize_initialization_catalog(state))
    return dict(state)


@pytest.mark.parametrize("case", ["unknown", "foreign", "wrong_label", "name_only", "predicate", "provenance"])
async def test_outline_supplied_invalid_assertion_never_disappears(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    from typing import cast

    from core.langgraph.state import NarrativeState

    provider = SyntheticSelector()
    state = cast(NarrativeState, await selected_state(tmp_path, monkeypatch, provider))
    catalog = select_catalog(state)
    assertion = {"source_id": catalog.candidates("Character")[0]["id"], "source_label": "Character", "target_id": catalog.candidates("Location")[0]["id"], "target_label": "Location", "relationship_type": "ORIGINATES_FROM", "description": "Born there."}
    invalid = dict(assertion)
    if case in {"unknown", "foreign", "name_only"}:
        invalid["target_id"] = {"unknown": "entity_unknown", "foreign": "entity_other_project_only", "name_only": "Ashgrove Weather Station"}[case]
    elif case == "wrong_label":
        invalid["target_label"] = "Item"
    elif case == "predicate":
        invalid["relationship_type"] = "NOT_A_PREDICATE"
    else:
        invalid["source_profile_managed"] = "true"
    provider.assertions = [assertion, invalid]
    with pytest.raises(ValueError):
        await extract_outline_relationships(state)
    assert state["outline_relationships_ref"] is None
    assert not (tmp_path / ".saga/initialization/selected").exists()
    assert len(provider.prompts) == 2


@pytest.mark.parametrize("selector", ["possessions", "participants", "location", "items"])
@pytest.mark.parametrize("case", ["positive", "empty", "null_role", "unknown", "wrong_label", "malformed"])
async def test_sibling_id_selectors_admit_all_or_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selector: str, case: str) -> None:
    from typing import cast

    from core.langgraph.state import NarrativeState

    provider = SyntheticSelector()
    state = cast(NarrativeState, await selected_state(tmp_path, monkeypatch, provider))
    catalog = select_catalog(state)
    character = catalog.candidates("Character")[0]["id"]
    item = catalog.candidates("Item")[0]["id"]
    location = catalog.candidates("Location")[0]["id"]
    role = None if case == "null_role" else "participant"
    if selector == "possessions":
        prefix, kind, field = "Catalog possessions", "POSSESSES", "item_id"
        positive: dict[str, Any] = {"character_id": character, "item_id": item}
        result: Any = {"possessions": [] if case == "empty" else [positive]}
    elif selector == "participants":
        prefix, kind, field = "Catalog event participants", "INVOLVES", "character_id"
        positive = {"character_id": character, "role": role}
        result = [] if case == "empty" else [positive]
    elif selector == "location":
        prefix, kind, field = "Catalog event location", "OCCURS_AT", "location_id"
        positive = {"location_id": None if case in {"empty", "null_role"} else location}
        result = positive
    else:
        prefix, kind, field = "Catalog event items", "FEATURES_ITEM", "item_id"
        positive = {"item_id": item, "role": role}
        result = {"featured_items": [] if case == "empty" else [positive]}
    if case == "unknown":
        positive[field] = "entity_unknown"
    elif case == "wrong_label":
        positive[field] = location if selector != "location" else item
    elif case == "malformed":
        positive["unadmitted"] = True
    provider.responses[prefix] = result
    state.update(await extract_outline_relationships(state))
    importer = InitializationImport(str(tmp_path))
    if case in {"unknown", "wrong_label", "malformed"}:
        with pytest.raises(ValueError):
            await importer.prepare(state)
        assert not (tmp_path / ".saga/initialization/selected").exists()
        return
    plan = await importer.prepare(state)
    act_ids = {entity.identity for entity in catalog.entities if entity.label == "Event" and json.loads(entity.payload)["event_type"] == "ActKeyEvent"}
    edges = [json.loads(statement.parameters) for statement in plan.statements if f"relationship:{kind}]" in statement.query]
    selected = [edge for edge in edges if edge["source"] == character] if selector == "possessions" else [edge for edge in edges if edge["source"] in act_ids]
    expected = 0 if case == "empty" or (selector == "location" and case == "null_role") else 1 if selector == "possessions" else len(act_ids)
    assert len(selected) == expected
    if case == "null_role":
        assert all("role" not in edge["properties"] for edge in selected)


@pytest.mark.parametrize("case", ["project", "parent", "catalog_checksum", "relationship_binding"])
async def test_catalog_binding_rejects_before_plan_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    from typing import cast

    from core.langgraph.state import NarrativeState

    provider = SyntheticSelector()
    state = cast(NarrativeState, await selected_state(tmp_path, monkeypatch, provider))
    state.update(await extract_outline_relationships(state))
    manager = ContentManager(str(tmp_path))
    if case == "project":
        state["graph_project_id"] = "00000000-0000-4000-8000-000000000001"
    elif case == "parent":
        assert state["character_sheets_ref"] is not None
        sheets = manager.load_json_strict(state["character_sheets_ref"])
        sheets["Ada"]["motivations"] = "Different accepted edit"
        state["character_sheets_ref"] = manager.save_json(sheets, "character_sheets", "all", version=3)
    elif case == "catalog_checksum":
        assert state["initialization_catalog_ref"] is not None
        path = tmp_path / state["initialization_catalog_ref"]["path"]
        path.write_text(path.read_text().replace("Brass Rain Gauge", "Glass Rain Gauge"))
    else:
        assert state["outline_relationships_ref"] is not None
        value = manager.load_json_strict(state["outline_relationships_ref"])
        value["catalog_identity"] = "0" * 64
        state["outline_relationships_ref"] = manager.save_json(value, "outline_relationships", "foreign", version=2)
    count = len(provider.prompts)
    with pytest.raises(ContentIntegrityError if case == "catalog_checksum" else ValueError):
        await InitializationImport(str(tmp_path)).prepare(state)
    assert len(provider.prompts) == count


async def test_duplicate_long_event_names_select_by_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SyntheticSelector()
    state = example_state(tmp_path)
    state["outline_relationships_ref"] = None
    manager = ContentManager(str(tmp_path))
    assert state["global_outline_ref"] is not None
    source = manager.load_json_strict(state["global_outline_ref"])
    source["inciting_incident"] = source["midpoint"] = "Ada visits the measuring station " * 20
    state["global_outline_ref"] = manager.save_json(source, "global_outline", "duplicate", version=2)
    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider)
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    state.update(await materialize_initialization_catalog(state))
    catalog = select_catalog(state)
    events = catalog.candidates("Event")[:2]
    assert events[0]["name"] == events[1]["name"] and events[0]["id"] != events[1]["id"]
    provider.assertions = [{"source_id": event["id"], "source_label": "Event", "target_id": catalog.candidates("Item")[0]["id"], "target_label": "Item", "relationship_type": "FEATURES_ITEM", "description": event["description"]} for event in events]
    state.update(await extract_outline_relationships(state))
    plan = await InitializationImport(str(tmp_path)).prepare(state)
    for event in events:
        assert any(json.loads(statement.parameters).get("source") == event["id"] and "relationship:FEATURES_ITEM" in statement.query for statement in plan.statements)


@pytest.mark.parametrize("stage", ["catalog", "relationships"])
async def test_selected_producer_acknowledgement_gap_never_reextracts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    provider = SyntheticSelector()
    state = example_state(tmp_path)
    state["outline_relationships_ref"] = None
    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider)
    if stage == "relationships":
        state.update(await materialize_initialization_catalog(state))
    node = materialize_initialization_catalog if stage == "catalog" else extract_outline_relationships
    selected = await node(state)
    calls = len(provider.prompts)
    assert await node(state) == selected
    assert len(provider.prompts) == calls


@pytest.mark.parametrize("version", [1, 2])
def test_snapshot_version_never_reinterprets_endpoint_schema(tmp_path: Path, version: int) -> None:
    from core.langgraph.initialization.snapshot import InitializationSnapshot, select_snapshot

    payload = select_snapshot(example_state(tmp_path)).model_dump(mode="json")
    payload["schema_version"] = version
    payload["relationships"] = [{
        **({"source_id": "character_a", "source_label": "Character", "target_id": "item_a", "target_label": "Item"} if version == 1 else {"source_name": "Ada", "target_name": "Rain Gauge"}),
        "relationship_type": "POSSESSES", "description": "Synthetic assertion",
    }]
    with pytest.raises(ValueError, match="relationship schema"):
        InitializationSnapshot.model_validate_json(json.dumps(payload))


async def test_unfrozen_v1_has_specific_actionable_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.langgraph.initialization.graph_plan import produce_plan
    from core.langgraph.initialization.snapshot import select_snapshot

    state = example_state(tmp_path)
    provider = AsyncMock()
    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider)
    for operation in (materialize_initialization_catalog(state), produce_plan(select_snapshot(state))):
        with pytest.raises(ValueError, match="Unfrozen legacy v1.*separate project"):
            await operation
    provider.assert_not_awaited()


async def test_selected_catalog_sqlite_resume_then_acceptance_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    from core.db_manager import Neo4jManagerSingleton
    from core.langgraph.workflow import create_checkpointer, create_full_workflow_graph

    provider = SyntheticSelector()
    state = example_state(tmp_path)
    state["outline_relationships_ref"] = None
    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider)
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    configuration = {"configurable": {"thread_id": "synthetic_catalog_recovery"}}
    database_path = str(tmp_path / "checkpoints/saga.db")
    async with create_checkpointer(database_path) as saver:
        graph = create_full_workflow_graph(saver)
        await graph.aupdate_state(configuration, state, as_node="init_all_chapter_outlines")
        graph.interrupt_after_nodes = ["init_catalog"]
        await graph.ainvoke(None, configuration)
        selected = await graph.aget_state(configuration)
        assert selected.next == ("init_outline_relationships",)
        assert selected.values["initialization_catalog_ref"]["checksum"]
    calls = len(provider.prompts)
    async with create_checkpointer(database_path) as saver:
        graph = create_full_workflow_graph(saver)
        graph.interrupt_after_nodes = ["init_commit_to_graph"]
        await graph.ainvoke(None, configuration)
        selected = await graph.aget_state(configuration)
        assert selected.next == ("init_persist_files",)
        importer = InitializationImport(str(tmp_path))
        plan = importer.load()
        assert plan.snapshot.schema_version == 2
        assert calls == 1
        assert sum("world_items_lines" in item.template for item in plan.evidence) == 1
        assert selected.values["initialization_id"] == plan.identity
    calls = len(provider.prompts)
    # Narrow transaction transport fault. This checks production sequencing/replay,
    # not Neo4j/APOC's engine rollback semantics (qualified separately).
    monkeypatch.setattr(Neo4jManagerSingleton, "_instance", None)
    database = Neo4jManagerSingleton()
    monkeypatch.setattr(get_services(), "database", database)
    attempts: list[list[tuple[str, Any]]] = []
    accepted: list[str] = []

    async def read_receipt(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"identity": accepted[0] if accepted else None}]

    async def transaction(callback: Any) -> None:
        statements: list[tuple[str, Any]] = []
        attempts.append(statements)

        def execute(query: str, parameters: Any = None, **keywords: Any) -> Any:
            result = MagicMock()
            if query == CONSTRAINT_QUERY:
                result.__iter__.return_value = iter(schema_catalog()[CONSTRAINT_QUERY])
            elif "RETURN owner.initialization_plan" in query:
                result.single.return_value = {"identity": None}
            elif "WHERE NOT n:SagaGraphOwner" in query:
                result.single.return_value = {"count": 0}
            elif parameters is not None:
                statements.append((query, parameters))
                if len(attempts) == 1 and len(statements) == 2:
                    raise OSError("synthetic transaction aborted")
            elif "RETURN count(n)" in query:
                result.single.return_value = {"count": 1}
            else:
                assert "SET owner.initialization_plan" in query
            return result

        callback(MagicMock(run=execute))
        accepted.append(plan.identity)

    monkeypatch.setattr(database, "execute_read_query", read_receipt)
    monkeypatch.setattr(database, "execute_in_transaction", transaction)
    with pytest.raises(OSError, match="transaction aborted"):
        await importer.accept(plan.identity)
    assert not accepted
    assert not (tmp_path / f".saga/initialization/{plan.identity}-accepted").exists()
    assert await InitializationImport(str(tmp_path)).accept(plan.identity) == plan
    assert attempts[1] == [(item.query, json.loads(item.parameters)) for item in plan.statements]
    (tmp_path / "characters/ada.yaml").write_text("user: deliberately edited after acceptance\n")
    assert await InitializationImport(str(tmp_path)).accept(plan.identity) == plan
    assert len(attempts) == 2
    assert len(provider.prompts) == calls
    assert (tmp_path / "characters/ada.yaml").read_text() == "user: deliberately edited after acceptance\n"
