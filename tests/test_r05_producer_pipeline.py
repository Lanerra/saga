"""From-scratch producer pipeline through frozen graph-plan admission (synthetic)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

import config
from core.graph_ownership import load_graph_project_id
from core.langgraph.content_manager import ContentManager
from core.langgraph.initialization.act_outlines_node import _get_act_role, generate_act_outlines
from core.langgraph.initialization.all_chapter_outlines_node import generate_all_chapter_outlines
from core.langgraph.initialization.catalog import materialize_initialization_catalog, select_catalog
from core.langgraph.initialization.character_sheets_node import generate_character_sheets
from core.langgraph.initialization.global_outline_node import generate_global_outline
from core.langgraph.initialization.outline_relationships_node import extract_outline_relationships
from core.langgraph.initialization.snapshot import select_snapshot
from core.langgraph.initialization.staged_import import InitializationImport
from core.langgraph.state import NarrativeState
from core.project_bootstrapper import ProjectBootstrapper
from core.project_config import allocate_word_target
from core.service_context import get_services
from tests.test_r05_initialization_contracts import chapter_response, global_response
from utils.text_processing import generate_entity_id


@pytest.mark.run_settings(GENERATE_ALL_CHAPTER_OUTLINES_AT_INIT=True, ENABLE_ENTITY_EMBEDDING_PERSISTENCE=False)
@pytest.mark.parametrize(("total", "act_count"), [(1, 1), (2, 2), (3, 3), (4, 3), (5, 3), (7, 3), (5, 5), (7, 5), (20, 3)])
async def test_actual_producers_admit_complete_frozen_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, total: int, act_count: int) -> None:
    phase = "bootstrap"
    characters = ["Ada", "Bea", "Cora", "Dara"]
    metadata = dict(title="Synthetic Harbor", genre="Mystery", theme="Duty", setting="Harbor", protagonist_name="Ada", narrative_style="Close third person", total_chapters=total, target_word_count=101)
    global_data = global_response(total, act_count)
    global_data["character_arcs"] = [dict(character_name=name, starting_state="Unsure", ending_state="Certain", key_moments=[f"{name} chooses"]) for name in characters]
    act_index = 0
    calls: list[dict[str, Any]] = []
    selected_world = [dict(name="Harbor", category="location", description="Cold water"), dict(name="Compass", category="object", description=" Brass\n")]

    async def transport(**arguments: Any) -> tuple[str, dict[str, Any]]:
        nonlocal act_index
        calls.append(arguments)
        prompt = arguments["prompt"]
        contract = arguments.get("response_format", {}).get("json_schema", {})
        if phase == "bootstrap":
            return json.dumps(metadata), {}
        if phase == "characters":
            if not contract:
                return json.dumps(characters), {}
            name = contract["schema"]["properties"]["name"]["enum"][0]
            return json.dumps(dict(name=name, description=f"{name} explores", traits=["brave"], status="Active", motivations=f"{name} protects Harbor", background="Carries Compass", skills=["navigation"], relationships={}, internal_conflict="Duty")), {}
        if phase == "global":
            assert contract["name"] == "global_outline"
            assert contract["strict"] is config.STRUCTURED_OUTPUT_STRICT
            assert arguments["auto_clean_response"] is False
            for name in characters:
                assert f"{name} protects Harbor" in prompt
            return json.dumps(global_data), {}
        if phase == "acts":
            act_index += 1
            number = act_index
            act = global_data["acts"][number - 1]
            assert contract["name"] == "act_outline"
            assert contract["schema"]["properties"]["act_number"]["enum"] == [number]
            sections: dict[str, Any] = {key: "Ada explores Harbor with Compass" for key in ("act_summary", "opening_situation", "character_development", "stakes_and_tension", "act_ending_turn", "thematic_thread", "pacing_notes")}
            sections["key_events"] = [dict(sequence=index, event=f"Ada explores {index}", cause="Choice", effect="Discovery") for index in range(1, 6)]
            return json.dumps(dict(act_number=number, total_acts=global_data["act_count"], act_role=_get_act_role(number, global_data["act_count"]), chapters_in_act=act["chapters_end"] - act["chapters_start"] + 1, sections=sections)), {}
        if phase == "chapters":
            assert contract["name"] == "chapter_outline"
            assert contract["strict"] is config.STRUCTURED_OUTPUT_STRICT
            assert arguments["auto_clean_response"] is False
            number_match = re.search(r"Chapter: (\d+) of", prompt)
            assert number_match is not None
            number = int(number_match[1])
            assert f"Chapter word target: {allocate_word_target(101, total, number)}" in prompt
            return json.dumps(chapter_response()), {}
        if phase == "catalog":
            return json.dumps(selected_world), {}
        if phase == "relationships":
            return '{"kg_triples": []}', {}
        assert phase == "plan"
        if contract["name"] == "catalog_event_characters":
            return json.dumps([dict(character_id=generate_entity_id("Ada", "character"), role="protagonist")]), {}
        if contract["name"] == "catalog_event_location":
            return json.dumps(dict(location_id=generate_entity_id("Harbor", "location"))), {}
        if contract["name"] == "catalog_event_items":
            return json.dumps(dict(featured_items=[dict(item_id=generate_entity_id("Compass", "object"), role=None)])), {}
        assert contract["name"] == "catalog_possessions"
        return json.dumps(dict(possessions=[dict(character_id=generate_entity_id("Ada", "character"), item_id=generate_entity_id("Compass", "object"))])), {}

    async def empty_traits(*arguments: Any, **keywords: Any) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(get_services().language_model, "async_call_llm", transport)
    monkeypatch.setattr(get_services().database, "execute_read_query", empty_traits)

    project = await ProjectBootstrapper(get_services().language_model).generate_metadata("Synthetic harbor mystery")
    state = cast(NarrativeState, {**project.model_dump(), "project_dir": str(tmp_path), "project_id": "synthetic", "graph_project_id": load_graph_project_id(tmp_path)})
    for phase, producer in [("characters", generate_character_sheets), ("global", generate_global_outline), ("acts", generate_act_outlines), ("chapters", generate_all_chapter_outlines), ("catalog", materialize_initialization_catalog), ("relationships", extract_outline_relationships)]:
        result = await producer(state)
        assert not result.get("last_error"), (phase, result)
        state.update(result)
    snapshot = select_snapshot(state)
    catalog = select_catalog(state)
    assert [sheet.name for sheet in snapshot.characters] == characters
    assert [chapter.chapter_number for chapter in snapshot.chapters] == list(range(1, total + 1))
    assert len([entity for entity in catalog.entities if entity.label == "Chapter"]) == total
    assert {json.loads(entity.payload)["name"] for entity in catalog.entities if entity.label in {"Item", "Location"}} == {item["name"] for item in selected_world}
    manager = ContentManager(str(tmp_path))
    source_bytes = {artifact.path: manager.load_text_strict(artifact.reference()) for artifact in snapshot.artifacts}
    phase = "plan"
    importer = InitializationImport(str(tmp_path))
    plan = await importer.prepare(state)
    assert plan.snapshot == snapshot
    assert plan.entities == catalog.entities
    assert importer.load() == plan
    restored = importer.state(plan)
    assert restored["target_word_count"] == metadata["target_word_count"]
    assert restored["narrative_style"] == metadata["narrative_style"]
    assert restored["protagonist_name"] == metadata["protagonist_name"]
    projections = json.loads(plan.projections)
    saga = yaml.safe_load(projections["saga.yaml"])
    assert saga["target_word_count"] == 101
    assert saga["narrative_style"] == metadata["narrative_style"]
    world = yaml.safe_load(projections["world/items.yaml"])
    assert [{key: item[key] for key in ("name", "category", "description")} for item in world["items"]] == selected_world
    outlines = yaml.safe_load(projections["outline/beats.yaml"])["selected_outlines"]
    assert outlines == {name: snapshot.source(name) for name in ("global_outline", "act_outlines", "chapter_outlines")}
    assert yaml.safe_load(projections["characters/ada.yaml"])["selected_sheet"] == snapshot.source("character_sheets")["Ada"]
    count = len(calls)
    assert await importer.prepare({**state, "initialization_id": plan.identity}) == plan
    assert len(calls) == count
    assert {artifact.path: manager.load_text_strict(artifact.reference()) for artifact in snapshot.artifacts} == source_bytes
    chapter_targets = [allocate_word_target(101, total, number) for number in range(1, total + 1)]
    assert sum(chapter_targets) == 101
    for target in chapter_targets:
        assert sum(allocate_word_target(target, 3, number) for number in range(1, 4)) == target


@pytest.mark.run_settings(GENERATE_ALL_CHAPTER_OUTLINES_AT_INIT=False)
async def test_disabled_upfront_outlines_fail_before_unsupported_admission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = await generate_all_chapter_outlines({"project_dir": str(tmp_path), "total_chapters": 2})
    assert result.get("has_fatal_error") is True
    assert result["initialization_step"] == "all_chapter_outlines_failed"
    assert "requires" in str(result["last_error"]).lower()
