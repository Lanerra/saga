"""Synthetic initialization admission and replay contracts."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from core.graph_ownership import load_graph_project_id
from core.langgraph.content_manager import ContentManager
from core.langgraph.initialization import all_chapter_outlines_node, commit_init_node
from core.langgraph.state import NarrativeState
from core.service_context import get_services
from utils.file_io import write_yaml_file


def example_state(tmp_path: Path) -> NarrativeState:
    manager = ContentManager(str(tmp_path))
    global_outline = {
        "act_count": 1, "acts": [{"act_number": 1, "title": "Journey", "summary": "Ada explores", "chapters_start": 1, "chapters_end": 1}],
        "inciting_incident": "Ada departs", "midpoint": "Ada discovers", "climax": "Ada decides", "resolution": "Ada returns",
        "character_arcs": [], "thematic_progression": "Courage", "raw_text": "Ada departs and returns.", "total_chapters": 1,
    }
    sections: dict[str, Any] = {name: "Ada explores." for name in (
        "act_summary", "opening_situation", "character_development", "stakes_and_tension", "act_ending_turn", "thematic_thread", "pacing_notes",
    )}
    sections["key_events"] = [{"sequence": number, "event": f"Ada explores {number}", "cause": "Choice", "effect": "Discovery"} for number in range(1, 6)]
    payloads: dict[str, Any] = {
        "character_sheets": {"Ada": {"name": "Ada", "description": "Explorer", "traits": ["brave"], "status": "Active", "motivations": "Discover", "background": "Harbor", "skills": ["navigation"], "internal_conflict": "Duty", "is_protagonist": True, "relationships": {}}},
        "global_outline": global_outline,
        "act_outlines": {"format_version": 2, "acts": [{"act_number": 1, "total_acts": 1, "act_role": "Journey", "chapters_in_act": 1, "sections": sections}]},
        "chapter_outlines": {"1": {"chapter_number": 1, "act_number": 1, "scene_description": "Ada explores", "key_beats": ["Ada chooses"], "plot_point": "Choice", "version": 0}},
        "outline_relationships": [],
    }
    state: dict[str, Any] = {"project_dir": str(tmp_path), "project_id": tmp_path.name, "graph_project_id": load_graph_project_id(tmp_path), "total_chapters": 1, "title": "Synthetic", "setting": "Harbor"}
    for name, payload in payloads.items():
        state[name + "_ref"] = manager.save_json(payload, name, "all", version=0 if name == "chapter_outlines" else 1)
    return cast(NarrativeState, state)


async def test_stage_is_frozen_before_any_graph_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = example_state(tmp_path)
    provider = AsyncMock(return_value=("[]", {}))
    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider)
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    writes = AsyncMock()
    monkeypatch.setattr(get_services().database, "execute_cypher_batch", writes)
    result = await commit_init_node.commit_initialization_to_graph(state)
    assert result["initialization_step"] == "initialization_prepared"
    assert writes.await_count == 0
    assert len(result["initialization_id"]) == 64
    count = provider.await_count
    replay = await commit_init_node.commit_initialization_to_graph({**state, **result})
    assert replay == result
    assert provider.await_count == count


async def test_missing_chapter_never_publishes_partial_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from config.settings import settings

    monkeypatch.setattr(settings, "GENERATE_ALL_CHAPTER_OUTLINES_AT_INIT", True)
    monkeypatch.setattr(all_chapter_outlines_node, "_determine_act_for_chapter", lambda state, number: 1)
    generator = AsyncMock(side_effect=[{"chapter_number": 1, "act_number": 1}, None, {"chapter_number": 3, "act_number": 1}])
    monkeypatch.setattr(all_chapter_outlines_node, "_generate_single_chapter_outline", generator)
    result = await all_chapter_outlines_node.generate_all_chapter_outlines({"project_dir": str(tmp_path), "total_chapters": 3})
    assert "chapter_outlines_ref" not in result
    assert result["has_fatal_error"] is True
    assert not (tmp_path / ".saga/content/chapter_outlines/all_v0.json").exists()


async def test_declared_corrupt_relationships_fail_before_domain_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ContentManager(str(tmp_path))
    characters = manager.save_json({"Ada": {"name": "Ada", "traits": ["brave"], "description": "Explorer"}}, "character_sheets", "all")
    outline = manager.save_json({"raw_text": "Ada explores."}, "global_outline", "main")
    relationships = manager.save_json([], "outline_relationships", "all")
    (tmp_path / relationships["path"]).write_text("[{}]", encoding="utf-8")
    writes = AsyncMock()
    monkeypatch.setattr(get_services().database, "execute_cypher_batch", writes)
    monkeypatch.setattr(commit_init_node, "_extract_world_items_from_outline", AsyncMock(return_value=[]))
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    result = await commit_init_node.commit_initialization_to_graph({
        "project_dir": str(tmp_path), "character_sheets_ref": characters,
        "global_outline_ref": outline, "outline_relationships_ref": relationships,
    })
    assert result.get("has_fatal_error") is True
    assert writes.await_count == 0


@pytest.mark.parametrize("case", ["chapter_gap", "chapter_identity", "version", "character_identity", "act_gap", "global_act_duplicate", "relationship_root", "relationship_item"])
async def test_semantic_admission_precedes_provider_and_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    state: dict[str, Any] = dict(example_state(tmp_path))
    manager = ContentManager(str(tmp_path))
    name = "chapter_outlines"
    value = manager.load_json_strict(state[name + "_ref"])
    if case == "chapter_gap":
        value = {}
    elif case == "chapter_identity":
        value["1"]["chapter_number"] = 2
    elif case == "version":
        value["1"]["version"] = 1
    elif case == "character_identity":
        name = "character_sheets"
        value = manager.load_json_strict(state[name + "_ref"])
        value["Ada"]["name"] = "Bea"
    elif case == "act_gap":
        name, value = "act_outlines", {"format_version": 2, "acts": []}
    elif case == "global_act_duplicate":
        name = "global_outline"
        value = manager.load_json_strict(state[name + "_ref"])
        value["acts"].append(value["acts"][0])
    elif case.startswith("relationship"):
        name = "outline_relationships"
        value = {} if case == "relationship_root" else [{}]
    reference = state[name + "_ref"]
    state[name + "_ref"] = manager.save_json(value, name, "admission", version=reference["version"])
    provider = AsyncMock(return_value=("[]", {}))
    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider)
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    result = await commit_init_node.commit_initialization_to_graph(cast(NarrativeState, state))
    assert result.get("has_fatal_error") is True
    assert provider.await_count == 0


async def test_projection_interruption_preserves_user_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.langgraph.initialization import persist_files_node
    from core.langgraph.initialization.staged_import import InitializationImport

    state = example_state(tmp_path)
    monkeypatch.setattr(get_services().language_model, "async_call_llm", AsyncMock(return_value=("[]", {})))
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    prepared = await commit_init_node.commit_initialization_to_graph(state)
    importer = InitializationImport(str(tmp_path))
    plan = importer.load()
    original = persist_files_node._publish_new_file
    count = 0

    def interrupted(path: Path, data: Any, *, writer: Callable[[Path, Any], None] = write_yaml_file) -> None:
        nonlocal count
        original(path, data, writer=writer)
        if path.is_relative_to(tmp_path) and not path.is_relative_to(tmp_path / ".saga"):
            count += 1
            if count == 2:
                raise OSError("synthetic projection interruption")

    monkeypatch.setattr(persist_files_node, "_publish_new_file", interrupted)
    with pytest.raises(OSError, match="synthetic projection interruption"):
        importer.publish_projections(plan)
    monkeypatch.setattr(persist_files_node, "_publish_new_file", original)
    importer.publish_projections(plan)
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*.yaml")}
    importer.publish_projections(plan)
    assert {str(path): path.read_bytes() for path in tmp_path.rglob("*.yaml")} == before
    rules = tmp_path / "world/rules.yaml"
    rules.write_text("rules: [user-owned]\n")
    with pytest.raises(ValueError, match="User-owned projection differs"):
        importer.publish_projections(plan)
    assert rules.read_text() == "rules: [user-owned]\n"
    assert prepared["initialization_id"] == plan.identity


async def test_duplicate_json_keys_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib

    state = example_state(tmp_path)
    assert state["chapter_outlines_ref"] is not None
    reference = state["chapter_outlines_ref"].copy()
    path = tmp_path / reference["path"]
    text = path.read_text().replace('"chapter_number": 1', '"chapter_number": 9, "chapter_number": 1')
    path.write_text(text)
    reference["size_bytes"] = len(text.encode())
    reference["checksum"] = hashlib.sha256(text.encode()).hexdigest()
    state["chapter_outlines_ref"] = reference
    monkeypatch.setattr(get_services().language_model, "async_call_llm", AsyncMock(return_value=("[]", {})))
    monkeypatch.setattr("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False)
    result = await commit_init_node.commit_initialization_to_graph(state)
    assert result.get("has_fatal_error") is True
