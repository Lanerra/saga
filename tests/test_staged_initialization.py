"""Synthetic initialization admission and replay contracts."""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from core.graph_ownership import load_graph_project_id
from core.langgraph.content_manager import ContentManager
from core.langgraph.initialization import all_chapter_outlines_node, commit_init_node
from core.langgraph.initialization.act_outlines_node import _get_act_role, generate_act_outlines
from core.langgraph.initialization.chapter_allocation import choose_act_ranges, determine_act_for_chapter
from core.langgraph.initialization.global_outline_node import _parse_global_outline, generate_global_outline
from core.langgraph.initialization.snapshot import select_snapshot
from core.langgraph.state import NarrativeState
from core.project_config import NarrativeProjectConfig
from core.service_context import get_services
from prompts.prompt_renderer import render_prompt
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


def with_catalog(state: NarrativeState, relationships: list[dict[str, Any]] | None = None) -> NarrativeState:
    """Real parser materialization with explicitly synthetic empty world and ID assertions."""
    from core.langgraph.initialization.catalog import materialize_entities
    from core.langgraph.initialization.snapshot import select_inputs

    manager = ContentManager(state["project_dir"])
    catalog = materialize_entities(select_inputs(state), [], ())
    assertions = {"schema_version": 2, "project_id": catalog.inputs.project_id, "catalog_identity": catalog.identity, "relationships": relationships or [], "evidence": []}
    return {
        **state,
        "initialization_catalog_ref": manager.save_json(catalog.model_dump(mode="json"), "initialization_catalog", catalog.inputs.identity, version=2),
        "outline_relationships_ref": manager.save_json(assertions, "outline_relationships", catalog.identity + "_fixture", version=2),
    }


async def test_stage_is_frozen_before_any_graph_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = with_catalog(example_state(tmp_path))
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

    state = with_catalog(example_state(tmp_path))
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


def rendered_global_prompt(total_chapters: int) -> str:
    project = NarrativeProjectConfig(
        title="Synthetic", genre="Adventure", theme="Courage", setting="Harbor",
        protagonist_name="Ada", narrative_style="Direct", total_chapters=total_chapters,
        target_word_count=1200,
    )
    return render_prompt("initialization/generate_global_outline.j2", {
        **project.model_dump(), "character_context": "Ada: explorer", "character_names": ["Ada"],
    })


def state_with_global_outline(tmp_path: Path, total_chapters: int, outline: dict[str, Any]) -> NarrativeState:
    state = example_state(tmp_path)
    state["total_chapters"] = total_chapters
    manager = ContentManager(str(tmp_path))
    assert state["act_outlines_ref"] is not None
    act_example = manager.load_json_strict(state["act_outlines_ref"])["acts"][0]
    ranges = choose_act_ranges(outline, total_chapters)
    acts = [{
        **act_example, "act_number": number, "total_acts": outline["act_count"],
        "act_role": _get_act_role(number, outline["act_count"]), "chapters_in_act": allocation.chapters_in_act,
    } for number, allocation in ranges.items()]
    chapters = {str(number): {
        "chapter_number": number, "act_number": determine_act_for_chapter(outline, total_chapters, number),
        "scene_description": "Ada explores", "key_beats": ["Ada chooses"], "plot_point": "Choice", "version": 0,
    } for number in range(1, total_chapters + 1)}
    state["global_outline_ref"] = manager.save_json(outline, "global_outline", "selected", version=1)
    state["act_outlines_ref"] = manager.save_json({"format_version": 2, "acts": acts}, "act_outlines", "selected", version=1)
    state["chapter_outlines_ref"] = manager.save_json(chapters, "chapter_outlines", "selected", version=0)
    return state


@pytest.mark.parametrize(("total_chapters", "allowed"), [(1, (1,)), (2, (2,)), (3, (3,)), (4, (3,)), (5, (3, 5)), (20, (3, 5))])
def test_global_prompt_only_requests_feasible_act_counts(total_chapters: int, allowed: tuple[int, ...]) -> None:
    prompt = rendered_global_prompt(total_chapters)
    constraint = next(line for line in prompt.splitlines() if line.startswith("- act_count must be"))
    assert tuple(int(value) for value in re.findall(r"\d+", constraint)) == allowed
    assert all(number <= total_chapters for number in allowed)


@pytest.mark.parametrize("total_chapters", [1, 2, 3, 4, 5, 20])
def test_rendered_outline_example_passes_production_parser_and_admission(tmp_path: Path, total_chapters: int) -> None:
    prompt = rendered_global_prompt(total_chapters)
    response = prompt.split("```json\n", 1)[1].split("```", 1)[0]
    outline = _parse_global_outline(response, {"total_chapters": total_chapters})
    assert outline["validation_errors"] == []
    snapshot = select_snapshot(state_with_global_outline(tmp_path, total_chapters, outline))
    assert snapshot.total_chapters == total_chapters
    assert snapshot.total_acts == min(total_chapters, 3)
    assert [chapter.chapter_number for chapter in snapshot.chapters] == list(range(1, total_chapters + 1))
    assert outline["structure_type"] == f"{snapshot.total_acts}-act"


@pytest.mark.parametrize("total_chapters", [5, 20])
def test_five_act_outline_still_passes_strict_admission(tmp_path: Path, total_chapters: int) -> None:
    response = rendered_global_prompt(total_chapters).split("```json\n", 1)[1].split("```", 1)[0]
    source = json.loads(response)
    ranges = choose_act_ranges({"act_count": 5}, total_chapters)
    source["act_count"] = 5
    source["acts"] = [{
        "act_number": number, "title": f"Act {number}", "summary": "Ada explores", "key_events": ["Ada chooses"],
        "chapters_start": allocation.chapters_start, "chapters_end": allocation.chapters_end,
    } for number, allocation in ranges.items()]
    source["character_arcs"] = []
    outline = _parse_global_outline(json.dumps(source), {"total_chapters": total_chapters})
    assert outline["validation_errors"] == []
    snapshot = select_snapshot(state_with_global_outline(tmp_path, total_chapters, outline))
    assert snapshot.total_acts == 5
    assert outline["structure_type"] == "5-act"


@pytest.mark.parametrize(("case", "message"), [("overlap", "partition chapter topology"), ("empty", "Malformed global act range")])
def test_selected_outline_invalid_partition_remains_rejected(tmp_path: Path, case: str, message: str) -> None:
    state = example_state(tmp_path)
    manager = ContentManager(str(tmp_path))
    assert state["global_outline_ref"] is not None
    outline = manager.load_json_strict(state["global_outline_ref"])
    if case == "overlap":
        outline["act_count"] = 2
        outline["acts"].append({**outline["acts"][0], "act_number": 2})
    else:
        outline["acts"][0]["chapters_start"] = 2
    state["global_outline_ref"] = manager.save_json(outline, "global_outline", "invalid", version=1)
    with pytest.raises(ValueError, match=message):
        select_snapshot(state)


@pytest.mark.parametrize(("total_acts", "roles"), [
    (1, ["Setup/Confrontation/Climax/Resolution"]),
    (2, ["Setup/Rising Action", "Resolution/Climax"]),
])
def test_short_act_prompt_roles_cover_the_complete_story(total_acts: int, roles: list[str]) -> None:
    for number, role in enumerate(roles, 1):
        actual_role = _get_act_role(number, total_acts)
        prompt = render_prompt("initialization/generate_act_outline.j2", {
            "title": "Synthetic", "genre": "Adventure", "theme": "Courage", "setting": "Harbor", "protagonist_name": "Ada",
            "act_number": number, "total_acts": total_acts, "act_role": actual_role, "chapters_in_act": 1,
            "global_outline": "Ada leaves and returns.", "character_context": "Ada: explorer",
        })
        assert actual_role == role
        assert f'"act_role" (str): MUST equal "{role}"' in prompt


@pytest.mark.parametrize("total_chapters", [1, 2, 3, 4, 5, 20])
async def test_generated_outline_nodes_reach_strict_admission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, total_chapters: int) -> None:
    from config.settings import settings

    seed = example_state(tmp_path / "seed")
    seed_manager = ContentManager(seed["project_dir"])
    assert seed["act_outlines_ref"] is not None
    assert seed["character_sheets_ref"] is not None
    sections = seed_manager.load_json_strict(seed["act_outlines_ref"])["acts"][0]["sections"]
    manager = ContentManager(str(tmp_path / "generated"))
    state: NarrativeState = {
        "project_dir": str(tmp_path / "generated"), "graph_project_id": load_graph_project_id(tmp_path / "generated"),
        "total_chapters": total_chapters, "title": "Synthetic", "protagonist_name": "Ada",
        "character_sheets_ref": manager.save_json(seed_manager.load_json_strict(seed["character_sheets_ref"]), "character_sheets", "all", version=1),
        "outline_relationships_ref": manager.save_json([], "outline_relationships", "all", version=1),
    }
    prompts: list[str] = []

    async def text_response(*, prompt: str, **keywords: Any) -> tuple[str, dict[str, int]]:
        prompts.append(prompt)
        if "## Chapter\n" in prompt:
            return json.dumps({"scene_description": "Ada explores", "key_beats": ["Ada chooses"], "plot_point": "Choice"}), {}
        return prompt.split("```json\n", 1)[1].split("```", 1)[0], {}

    async def act_response(*, prompt: str, **keywords: Any) -> tuple[dict[str, Any], dict[str, int]]:
        prompts.append(prompt)
        identity = re.search(r"Act: (\d+) of (\d+)", prompt)
        allocation = re.search(r"Chapters in act: ~(\d+)", prompt)
        role = re.search(r"Role in structure: (.+)", prompt)
        assert identity and allocation and role
        return {"act_number": int(identity[1]), "total_acts": int(identity[2]), "act_role": role[1], "chapters_in_act": int(allocation[1]), "sections": sections}, {}

    monkeypatch.setattr(settings, "GENERATE_ALL_CHAPTER_OUTLINES_AT_INIT", True)
    monkeypatch.setattr(get_services().language_model, "async_call_llm", text_response)
    monkeypatch.setattr(get_services().language_model, "async_call_llm_json_object", act_response)
    for node in (generate_global_outline, generate_act_outlines, all_chapter_outlines_node.generate_all_chapter_outlines):
        update = await node(state)
        assert update.get("last_error") is None
        state = {**state, **update}
    selected = select_snapshot(state)
    assert selected.total_chapters == total_chapters
    assert selected.total_acts == min(total_chapters, 3)
    assert len(prompts) == 1 + selected.total_acts + total_chapters
    chapter_prompts = [prompt for prompt in prompts if "## Chapter\n" in prompt]
    assert len(chapter_prompts) == total_chapters
    assert all("(Chapter 0 of this act)" not in prompt for prompt in chapter_prompts)


@pytest.mark.parametrize("metadata", [
    {"chapters_start": 1}, {"chapters_end": 1},
    {"chapters_start": True, "chapters_end": 1}, {"chapters_start": 1, "chapters_end": 1.0},
    {"chapters_start": 2, "chapters_end": 2}, {"chapters_start": 2, "chapters_end": 1},
])
def test_act_allocation_metadata_must_match_global_range(tmp_path: Path, metadata: dict[str, Any]) -> None:
    state = example_state(tmp_path)
    manager = ContentManager(str(tmp_path))
    assert state["act_outlines_ref"] is not None
    acts = manager.load_json_strict(state["act_outlines_ref"])
    acts["acts"][0].update(metadata)
    state["act_outlines_ref"] = manager.save_json(acts, "act_outlines", "invalid", version=1)
    with pytest.raises(ValueError, match="Act range metadata mismatch"):
        select_snapshot(state)
