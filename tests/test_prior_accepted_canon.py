"""Accepted history uses retained facts, never mutable chapter projections."""

from itertools import permutations
from pathlib import Path
from typing import Any, cast

import pytest

from core.db_manager import neo4j_manager
from core.langgraph.chapter_lifecycle import ChapterLifecycle, canonical_bytes, extraction_binding
from core.langgraph.content_manager import ContentManager
from core.langgraph.state import NarrativeState
from core.langgraph.subgraphs import validation
from tests.fakes.quality import example_quality_state
from tests.test_langgraph.test_chapter_lifecycle import example_state


def accepted_row(
    directory: Path, chapter: int, relationship_type: str = "HATES", traits: list[str] | None = None,
    *, channel: str = "standalone", entity_type: str = "Character",
    additional_relationships: tuple[dict[str, Any], ...] = (),
    empty_extraction: bool = False,
) -> tuple[NarrativeState, dict[str, Any]]:
    state = example_state(directory)
    state["current_chapter"] = chapter
    manager = ContentManager(str(directory))
    relationships = [{"source_name": "Alice", "target_name": "Bob", "relationship_type": relationship_type, "description": "Synthetic", "chapter": chapter}]
    profile = {"Bob": {"type": relationship_type, "description": "Synthetic", "target_label": "Character"}}
    entity = {"name": "Alice", "type": entity_type, "attributes": {"traits": traits or [], "relationships": profile if channel != "standalone" else {}}}
    if channel == "profile":
        relationships = []
    relationships.extend(additional_relationships)
    if channel == "repeated":
        relationships *= 2
    entities = [entity, entity] if channel == "repeated" else [entity]
    if empty_extraction:
        relationships = []
        entities = []
    state["extracted_relationships_ref"] = manager.save_json(relationships, "relationships", f"chapter_{chapter}")
    state["extracted_entities_ref"] = manager.save_json(
        {"characters": entities if entity_type == "Character" else [], "world_items": entities if entity_type != "Character" else []}, "entities", f"chapter_{chapter}",
    )
    state["extraction_outcomes"] = [{**slot, "chapter_number": chapter} for slot in state["extraction_outcomes"]]
    scene_reference = state["scene_drafts_ref"]
    assert scene_reference is not None
    state["extraction_source"] = extraction_binding(state, manager.load_list_of_texts(scene_reference))
    state = example_quality_state(state)
    lifecycle = ChapterLifecycle(state).stage()
    acceptance = lifecycle._acceptance()
    return state, {"id": lifecycle.manifest.attempt_id, "manifest": lifecycle.manifest.encoded.decode(), "phase": "accepted", "acceptance": canonical_bytes(acceptance).decode(), "chapter": chapter, "chapter_attempt_id": lifecycle.manifest.attempt_id, "chapter_project_id": state["graph_project_id"], "chapter_status": "finalized"}


def accepted_empty_prefix(directory: Path, stop_chapter: int) -> list[dict[str, Any]]:
    return [accepted_row(directory, chapter, empty_extraction=True)[1] for chapter in range(1, stop_chapter)]


@pytest.mark.parametrize("entrypoint", ["_fetch_validation_data", "detect_contradictions", "validate_consistency"])
@pytest.mark.parametrize("missing_chapter", [1, 2, 3])
async def test_incomplete_accepted_prefix_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_chapter: int, entrypoint: str) -> None:
    from core.langgraph.nodes.validation_node import validate_consistency

    rows = accepted_empty_prefix(tmp_path, 4)
    state = example_state(tmp_path)
    state["current_chapter"] = 4
    rows = [row for row in rows if row["chapter"] != missing_chapter]
    monkeypatch.setattr(neo4j_manager, "require_project_binding", lambda: state["graph_project_id"])
    async def read(query: str, parameters: Any) -> list[dict[str, Any]]:
        return rows
    monkeypatch.setattr(neo4j_manager, "execute_read_query", read)
    consumer = validate_consistency if entrypoint == "validate_consistency" else getattr(validation, entrypoint)
    before = canonical_bytes(state)
    with pytest.raises(ValueError, match=rf"Prior accepted canon requires reconciliation: missing verified acceptance for chapters \[{missing_chapter}\]"):
        await consumer(state)
    assert canonical_bytes(state) == before


@pytest.mark.parametrize("current_chapter", [1, 4])
async def test_complete_empty_accepted_prefix_is_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, current_chapter: int) -> None:
    rows = accepted_empty_prefix(tmp_path, current_chapter)
    state = example_state(tmp_path)
    state["current_chapter"] = current_chapter
    monkeypatch.setattr(neo4j_manager, "require_project_binding", lambda: state["graph_project_id"])
    async def read(query: str, parameters: Any) -> list[dict[str, Any]]:
        return rows
    monkeypatch.setattr(neo4j_manager, "execute_read_query", read)
    assert await validation._fetch_validation_data(state) == {"relationships": {}, "characters": {}}


async def test_prior_acceptance_survives_current_row_permutations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = accepted_empty_prefix(tmp_path, 4)
    state, prior = accepted_row(tmp_path, 4)
    _, current = accepted_row(tmp_path, 5, "LOVES")
    state = cast(NarrativeState, {**state, "current_chapter": 5})
    monkeypatch.setattr(neo4j_manager, "require_project_binding", lambda: state["graph_project_id"])
    findings = []
    for order in permutations([prior, current]):
        async def read(query: str, parameters: Any, order: tuple[dict[str, Any], ...] = order) -> list[dict[str, Any]]:
            return [*prefix, *order]
        monkeypatch.setattr(neo4j_manager, "execute_read_query", read)
        history = await validation._fetch_validation_data(state)
        issues = await validation._check_relationship_evolution([{"source_name": "Alice", "target_name": "Bob", "relationship_type": "LOVES"}], 5, history["relationships"])
        findings.append([issue.model_dump() for issue in issues])
    assert [len(items) for items in findings] == [1, 1]
    assert findings[0] == findings[1]
    assert findings[0][0]["conflicting_chapters"] == [4, 5]


@pytest.mark.parametrize("phase", ["committed", "compensated", "planned", "staged", "rejected"])
async def test_unaccepted_attempts_are_not_truth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    prefix = accepted_empty_prefix(tmp_path, 4)
    state, row = accepted_row(tmp_path, 4)
    row["phase"] = phase
    state["current_chapter"] = 5
    monkeypatch.setattr(neo4j_manager, "require_project_binding", lambda: state["graph_project_id"])
    async def read(query: str, parameters: Any) -> list[dict[str, Any]]:
        return [*prefix, row]
    monkeypatch.setattr(neo4j_manager, "execute_read_query", read)
    with pytest.raises(ValueError, match=r"missing verified acceptance for chapters \[4\]"):
        await validation._fetch_validation_data(state)


@pytest.mark.parametrize("field,value", [("id", None), ("id", "invalid"), ("manifest", "{}"), ("acceptance", None), ("chapter_attempt_id", None), ("chapter_project_id", "foreign"), ("chapter_status", "generated")])
@pytest.mark.parametrize("channel", ["standalone", "profile", "mixed", "repeated"])
async def test_malformed_acceptance_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: Any, channel: str) -> None:
    prefix = accepted_empty_prefix(tmp_path, 4)
    state, row = accepted_row(tmp_path, 4, channel=channel)
    row[field] = value
    state["current_chapter"] = 5
    monkeypatch.setattr(neo4j_manager, "require_project_binding", lambda: state["graph_project_id"])
    async def read(query: str, parameters: Any) -> list[dict[str, Any]]:
        return [*prefix, row]
    monkeypatch.setattr(neo4j_manager, "execute_read_query", read)
    with pytest.raises((ValueError, TypeError)):
        await validation._fetch_validation_data(state)


async def test_history_failure_is_not_empty_canon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = example_state(tmp_path)
    monkeypatch.setattr(neo4j_manager, "require_project_binding", lambda: state["graph_project_id"])
    async def read(query: str, parameters: Any) -> list[dict[str, Any]]:
        raise RuntimeError("Synthetic read failure")
    monkeypatch.setattr(neo4j_manager, "execute_read_query", read)
    with pytest.raises(RuntimeError, match="Synthetic read failure"):
        await validation._fetch_validation_data(state)


async def test_trait_node_consumes_prior_accepted_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.langgraph.nodes.validation_node import validate_consistency

    prefix = accepted_empty_prefix(tmp_path, 4)
    state, row = accepted_row(tmp_path, 4, traits=["brave"])
    state["current_chapter"] = 5
    manager = ContentManager(str(tmp_path))
    state["extracted_entities_ref"] = manager.save_json({"characters": [{"name": "Alice", "attributes": {"traits": ["cowardly"]}}], "world_items": []}, "entities", "current")
    monkeypatch.setattr(neo4j_manager, "require_project_binding", lambda: state["graph_project_id"])
    async def read(query: str, parameters: Any) -> list[dict[str, Any]]:
        return [*prefix, row]
    monkeypatch.setattr(neo4j_manager, "execute_read_query", read)
    result = await validate_consistency(state)
    assert [(item.type, item.conflicting_chapters) for item in result["contradictions"]] == [("character_trait", [4, 5]), ("plot_stagnation", [5])]
    assert result["needs_revision"] is True


async def test_latest_prior_relationship_ties_are_order_independent() -> None:
    history = [
        {"rel_type": "HATES", "first_chapter": 1},
        {"rel_type": "HATES", "first_chapter": 4},
        {"rel_type": "LOVES", "first_chapter": 4},
        {"rel_type": "LOVES", "first_chapter": 5},
    ]
    findings = []
    candidate = {"source_name": "Alice", "target_name": "Bob", "relationship_type": "LOVES"}
    for order in permutations(history):
        issues = await validation._check_relationship_evolution([candidate, candidate], 5, {("Alice", "Bob"): list(order)})
        findings.append([issue.model_dump() for issue in issues])
    assert all(items == findings[0] for items in findings)
    assert len(findings[0]) == 1
    assert findings[0][0]["conflicting_chapters"] == [4, 5]


@pytest.mark.parametrize("barrier", ["compensation_required", "revision_rollback_failure", "extraction_failed"])
async def test_validation_preserves_failure_barriers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, barrier: str) -> None:
    state, row = accepted_row(tmp_path, 4)
    state["attempt_id"] = row["id"]
    if barrier == "compensation_required":
        ChapterLifecycle(state).load(row["id"]).observe("compensation_required")
    elif barrier == "revision_rollback_failure":
        state["revision_rollback_failure"] = cast(Any, {"error": "synthetic failed rollback"})
    else:
        state["extraction_status"] = "failed"
    calls = []
    async def read(query: str, parameters: Any) -> list[dict[str, Any]]:
        calls.append(query)
        return []
    monkeypatch.setattr(neo4j_manager, "require_project_binding", lambda: state["graph_project_id"])
    monkeypatch.setattr(neo4j_manager, "execute_read_query", read)
    with pytest.raises(ValueError):
        await validation.detect_contradictions(state)
    assert calls == []


async def test_duplicate_accepted_attempt_is_ambiguous(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = accepted_empty_prefix(tmp_path, 4)
    state, row = accepted_row(tmp_path, 4)
    state["current_chapter"] = 5
    async def read(query: str, parameters: Any) -> list[dict[str, Any]]:
        return [*prefix, row, row]
    monkeypatch.setattr(neo4j_manager, "require_project_binding", lambda: state["graph_project_id"])
    monkeypatch.setattr(neo4j_manager, "execute_read_query", read)
    with pytest.raises(ValueError, match="Ambiguous accepted"):
        await validation._fetch_validation_data(state)


async def test_retained_acceptance_mismatch_is_not_empty_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = accepted_empty_prefix(tmp_path, 4)
    state, row = accepted_row(tmp_path, 4)
    row["acceptance"] = "{}"
    state["current_chapter"] = 5
    async def read(query: str, parameters: Any) -> list[dict[str, Any]]:
        return [*prefix, row]
    monkeypatch.setattr(neo4j_manager, "require_project_binding", lambda: state["graph_project_id"])
    monkeypatch.setattr(neo4j_manager, "execute_read_query", read)
    with pytest.raises(ValueError, match="Graph/local acceptance mismatch"):
        await validation._fetch_validation_data(state)


@pytest.mark.parametrize("entity_type", ["Character", "Location", "Item", "Event"])
async def test_accepted_relationship_channels_are_equivalent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entity_type: str) -> None:
    findings = []
    for channel in ("standalone", "profile", "mixed", "repeated"):
        prefix = accepted_empty_prefix(tmp_path / channel, 4)
        state, row = accepted_row(tmp_path / channel, 4, traits=["brave"], channel=channel, entity_type=entity_type)
        state["current_chapter"] = 5
        manager = ContentManager(str(tmp_path / channel))
        state["extracted_relationships_ref"] = manager.save_json(
            [{"source_name": "Alice", "target_name": "Bob", "relationship_type": "LOVES", "description": "Synthetic", "chapter": 5}], "relationships", "candidate",
        )
        monkeypatch.setattr(neo4j_manager, "require_project_binding", lambda state=state: state["graph_project_id"])
        async def read(query: str, parameters: Any, row: dict[str, Any] = row, prefix: list[dict[str, Any]] = prefix) -> list[dict[str, Any]]:
            return [*prefix, row]
        monkeypatch.setattr(neo4j_manager, "execute_read_query", read)
        history = await validation._fetch_validation_data(state)
        assert history["relationships"] == {("Alice", "Bob"): [{"rel_type": "HATES", "first_chapter": 4}]}
        assert history["characters"] == ({"Alice": [{"traits": ["brave"], "first_chapter": 4}]} if entity_type == "Character" else {})
        output = await validation.detect_contradictions(state)
        findings.append([issue.model_dump() for issue in output["contradictions"] if issue.type == "relationship"])
    assert [len(items) for items in findings] == [1, 1, 1, 1]
    assert findings == [findings[0]] * 4


@pytest.mark.parametrize("entity_type", ["Character", "Location", "Item", "Event"])
@pytest.mark.parametrize("phase", ["current", "committed", "compensated", "planned", "staged", "rejected"])
async def test_profile_history_requires_prior_acceptance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entity_type: str, phase: str) -> None:
    prefix = accepted_empty_prefix(tmp_path, 4)
    state, row = accepted_row(tmp_path, 4, channel="profile", entity_type=entity_type)
    state["current_chapter"] = 4 if phase == "current" else 5
    if phase != "current":
        row["phase"] = phase
    monkeypatch.setattr(neo4j_manager, "require_project_binding", lambda: state["graph_project_id"])
    async def read(query: str, parameters: Any) -> list[dict[str, Any]]:
        return [*prefix, row]
    monkeypatch.setattr(neo4j_manager, "execute_read_query", read)
    if phase == "current":
        assert await validation._fetch_validation_data(state) == {"relationships": {}, "characters": {}}
    else:
        with pytest.raises(ValueError, match=r"missing verified acceptance for chapters \[4\]"):
            await validation._fetch_validation_data(state)


async def test_profile_history_order_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = accepted_empty_prefix(tmp_path, 3)
    state, earlier = accepted_row(tmp_path, 3, "DISTRUSTS", channel="profile")
    _, later = accepted_row(tmp_path, 4, channel="repeated")
    state["current_chapter"] = 5
    monkeypatch.setattr(neo4j_manager, "require_project_binding", lambda: state["graph_project_id"])
    histories = []
    for order in permutations([earlier, later]):
        async def read(query: str, parameters: Any, order: tuple[dict[str, Any], ...] = order) -> list[dict[str, Any]]:
            return [*prefix, *order]
        monkeypatch.setattr(neo4j_manager, "execute_read_query", read)
        histories.append(await validation._fetch_validation_data(state))
    expected = {"relationships": {("Alice", "Bob"): [{"rel_type": "DISTRUSTS", "first_chapter": 3}, {"rel_type": "HATES", "first_chapter": 4}]}, "characters": {}}
    assert histories == [expected, expected]


async def test_legacy_chapter_only_history_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    async def read(query: str, parameters: Any) -> list[dict[str, Any]]:
        calls.append(query)
        return []
    monkeypatch.setattr(neo4j_manager, "execute_read_query", read)
    with pytest.raises(ValueError, match="verified NarrativeState"):
        await validation._fetch_validation_data(2)
    assert calls == []


@pytest.mark.parametrize("entity_type", ["Character", "Location", "Item", "Event"])
@pytest.mark.parametrize("prior_channel", ["standalone", "profile", "mixed", "repeated"])
async def test_candidate_assertion_channels_are_equivalent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entity_type: str, prior_channel: str) -> None:
    prefix = accepted_empty_prefix(tmp_path, 4)
    state, row = accepted_row(tmp_path, 4, channel=prior_channel, entity_type=entity_type)
    state["current_chapter"] = 5
    manager = ContentManager(str(tmp_path))
    relationship = {"source_name": "Alice", "target_name": "Bob", "relationship_type": "LOVES", "description": "Synthetic", "chapter": 5}
    profile = {"name": "Alice", "type": entity_type, "attributes": {"relationships": {"Bob": {"type": " loves "}}}}
    monkeypatch.setattr(neo4j_manager, "require_project_binding", lambda: state["graph_project_id"])
    async def read(query: str, parameters: Any) -> list[dict[str, Any]]:
        return [*prefix, row]
    monkeypatch.setattr(neo4j_manager, "execute_read_query", read)
    findings = []
    for channel in ("standalone", "profile", "mixed", "repeated"):
        profiles = [] if channel == "standalone" else [profile] * (2 if channel == "repeated" else 1)
        state["extracted_entities_ref"] = manager.save_json(
            {"characters": profiles if entity_type == "Character" else [], "world_items": profiles if entity_type != "Character" else []}, "entities", channel,
        )
        state["extracted_relationships_ref"] = manager.save_json([] if channel == "profile" else [relationship] * (2 if channel == "repeated" else 1), "relationships", channel)
        before = canonical_bytes(state)
        output = await validation.detect_contradictions(state)
        findings.append([item.model_dump() for item in output["contradictions"]])
        assert canonical_bytes(state) == before
        assert (await validation._fetch_validation_data(state))["relationships"] == {("Alice", "Bob"): [{"first_chapter": 4, "rel_type": "HATES"}]}
    assert [len(items) for items in findings] == [1, 1, 1, 1]
    assert findings == [findings[0]] * 4
    assert findings[0][0]["conflicting_chapters"] == [4, 5]


@pytest.mark.parametrize("entrypoint", ["detect_contradictions", "validate_consistency"])
@pytest.mark.parametrize("profiles", [
    None, [], {"characters": None}, {"world_items": {}}, {"characters": [None]},
    {"characters": [{"name": " "}]}, {"characters": [{"name": "Alice", "attributes": []}]},
    {"characters": [{"name": "Alice", "attributes": {"relationships": []}}]},
    {"world_items": [{"name": "Alice", "attributes": {"relationships": {"": {"type": "LOVES"}}}}]},
    {"world_items": [{"name": "Alice", "attributes": {"relationships": {"Bob": "LOVES"}}}]},
    {"world_items": [{"name": "Alice", "attributes": {"relationships": {"Bob": {}}}}]},
    {"world_items": [{"name": "Alice", "attributes": {"relationships": {"Bob": {"type": " "}}}}]},
    {"world_items": [{"name": "Alice", "attributes": {"relationships": {"Bob": {"type": 1}}}}]},
])
async def test_malformed_candidate_profiles_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profiles: Any, entrypoint: str) -> None:
    prefix = accepted_empty_prefix(tmp_path, 4)
    state, row = accepted_row(tmp_path, 4)
    state["current_chapter"] = 5
    state["extracted_entities_ref"] = ContentManager(str(tmp_path)).save_json(profiles, "entities", "malformed")
    monkeypatch.setattr(neo4j_manager, "require_project_binding", lambda: state["graph_project_id"])
    async def read(query: str, parameters: Any) -> list[dict[str, Any]]:
        return [*prefix, row]
    monkeypatch.setattr(neo4j_manager, "execute_read_query", read)
    with pytest.raises(ValueError):
        await getattr(validation, entrypoint)(state)


async def test_candidate_profile_order_and_repetition_do_not_change_findings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = accepted_empty_prefix(tmp_path, 4)
    state, row = accepted_row(tmp_path, 4, "FEARS", channel="profile", additional_relationships=(
        {"source_name": "Zoe", "target_name": "Bob", "relationship_type": "HATES", "description": "Synthetic", "chapter": 4},
    ))
    state["current_chapter"] = 5
    manager = ContentManager(str(tmp_path))
    profiles = [
        {"name": "Alice", "type": "Character", "attributes": {"relationships": {"Bob": {"type": "PROTECTS"}}}},
        {"name": "Zoe", "type": "Event", "attributes": {"relationships": {"Bob": {"type": "LOVES"}}}},
    ]
    standalone = [
        {"source_name": "Alice", "target_name": "Bob", "relationship_type": "PROTECTS"},
        {"source_name": "Zoe", "target_name": "Bob", "relationship_type": "LOVES"},
    ]
    monkeypatch.setattr(neo4j_manager, "require_project_binding", lambda: state["graph_project_id"])
    async def read(query: str, parameters: Any) -> list[dict[str, Any]]:
        return [*prefix, row]
    monkeypatch.setattr(neo4j_manager, "execute_read_query", read)
    findings = []
    for number, (first, second) in enumerate(permutations(profiles)):
        state["extracted_entities_ref"] = manager.save_json({"characters": [first, second, first], "world_items": [second]}, "entities", str(number))
        for index, order in enumerate(permutations(standalone)):
            state["extracted_relationships_ref"] = manager.save_json(list(order) * 2, "relationships", f"{number}-{index}")
            output = await validation.detect_contradictions(state)
            findings.append([item.model_dump() for item in output["contradictions"]])
    assert [len(items) for items in findings] == [2, 2, 2, 2]
    assert findings == [findings[0]] * 4
    assert [item["description"].split(" and ")[0] for item in findings[0]] == ["Alice", "Zoe"]
