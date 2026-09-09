# tests/core/langgraph/subgraphs/test_scene_extraction_subgraph.py
import json
from pathlib import Path
from typing import Any

import pytest

from core.langgraph.content_manager import ContentManager
from core.langgraph.state import create_initial_state
from core.service_context import get_services
from tests.fakes.service_context import patch_service


@pytest.mark.asyncio
async def test_scene_extraction_subgraph_runs_extraction_and_consolidation(tmp_path: Path) -> None:
    from core.langgraph.subgraphs.scene_extraction import (
        create_scene_extraction_subgraph,
    )

    workflow = create_scene_extraction_subgraph()

    project_dir = str(tmp_path / "test_project")
    state = create_initial_state(
        project_id="test",
        title="Test Novel",
        genre="Fantasy",
        theme="Adventure",
        setting="World",
        target_word_count=50000,
        total_chapters=10,
        project_dir=project_dir,
        protagonist_name="Elara",
    )

    content_manager = ContentManager(project_dir)
    scenes = ["Elara enters the library.", "She finds the map."]
    state["scene_drafts_ref"] = content_manager.save_list_of_texts(scenes, "scenes", "chapter_1", 1)
    state["current_chapter"] = 1

    async def mock_llm(*args: object, **kwargs: object) -> tuple[dict[str, object], None]:
        return {"character_updates": {}, "world_updates": {"Location": {}, "Event": {}}, "kg_triples": []}, None

    with patch_service(
        'language_model.async_call_llm_json_object',
        side_effect=mock_llm,
    ):
        result = await workflow.ainvoke(state)

    assert "extracted_entities_ref" in result
    assert "extracted_relationships_ref" in result


EXTRACTION_TYPES = ("characters", "locations", "events", "relationships")
EMPTY_RESPONSES: tuple[dict[str, Any], ...] = (
    {"character_updates": {}},
    {"world_updates": {"Location": {}}},
    {"world_updates": {"Event": {}}},
    {"kg_triples": []},
)


@pytest.fixture
def extraction_state(tmp_path: Path) -> Any:
    state = create_initial_state(
        project_id="synthetic", title="Synthetic", genre="Fantasy", theme="Discovery",
        setting="Library", target_word_count=1000, total_chapters=1,
        project_dir=str(tmp_path), protagonist_name="Elara",
    )
    manager = ContentManager(str(tmp_path))
    state["scene_drafts_ref"] = manager.save_list_of_texts(
        ["Elara enters the Library.", "Elara leaves the Library."], "scenes", "chapter_1", 1,
    )
    state["chapter_plan_scene_count"] = 2
    return state


@pytest.mark.parametrize("failed_scene", [0, 1])
@pytest.mark.parametrize("failed_type", range(4), ids=EXTRACTION_TYPES)
@pytest.mark.parametrize("failure", ["invalid_json", "missing", "malformed", "invalid_item", "model_error"])
async def test_failed_slot_blocks_publication(
    extraction_state: Any, monkeypatch: pytest.MonkeyPatch,
    failed_scene: int, failed_type: int, failure: str,
) -> None:
    from core.exceptions import LLMServiceError
    from core.langgraph.nodes import scene_extraction
    from core.langgraph.subgraphs.scene_extraction import create_scene_extraction_subgraph

    assert Path(scene_extraction.__file__).resolve().parents[3] == Path(__file__).resolve().parents[4]
    import config

    monkeypatch.setitem(vars(config), "settings", config.settings.model_copy(update={"ENABLE_ENTITY_VALIDATION": False}))
    manager = ContentManager(extraction_state["project_dir"])
    extraction_state["extracted_entities_ref"] = manager.save_json(
        {"characters": [{"name": "stale"}], "world_items": []}, "extracted_entities", "chapter_1", 1,
    )
    extraction_state["extracted_relationships_ref"] = manager.save_json([], "extracted_relationships", "chapter_1", 1)
    malformed: tuple[dict[str, Any], ...] = ({"character_updates": []}, {"world_updates": {"Location": []}},
                 {"world_updates": {"Event": []}}, {"kg_triples": {}})
    invalid_items: tuple[dict[str, Any], ...] = ({"character_updates": {"Elara": None}}, {"world_updates": {"Location": {"Library": None}}},
                     {"world_updates": {"Event": {"Arrival": None}}}, {"kg_triples": [{}]})
    calls: list[int] = []
    slot = 0
    retried = False

    async def completion(**arguments: Any) -> tuple[str, None]:
        nonlocal slot, retried
        calls.append(slot)
        current = slot
        is_failure = current == failed_scene * 4 + failed_type
        if is_failure and failure == "invalid_json" and not retried:
            retried = True
            return "not JSON", None
        slot += 1
        if is_failure:
            if failure == "model_error":
                raise LLMServiceError("synthetic exhausted provider")
            if failure == "invalid_json":
                return "not JSON", None
            if failure == "missing":
                return "{}", None
            if failure == "malformed":
                return json.dumps(malformed[failed_type]), None
            return json.dumps(invalid_items[failed_type]), None
        return json.dumps(EMPTY_RESPONSES[current % 4]), None

    monkeypatch.setattr(get_services().language_model, "async_call_llm", completion)
    workflow = create_scene_extraction_subgraph()
    updates = [update async for update in workflow.astream(extraction_state, stream_mode="updates")]
    assert [list(update) for update in updates] == [["extract_from_scenes"]]
    result = updates[0]["extract_from_scenes"]
    assert result["has_fatal_error"] is True
    assert result["extraction_status"] == "failed"
    assert result["extraction_policy"] == "fail_closed"
    assert result["extracted_entities_ref"] is None
    assert result["extracted_relationships_ref"] is None
    assert result["error_node"] == "extract_from_scenes"
    outcomes = result["extraction_outcomes"]
    assert [(entry["scene_index"], entry["extraction_type"]) for entry in outcomes] == [
        (scene, kind) for scene in range(2) for kind in EXTRACTION_TYPES
    ]
    assert [entry["status"] for entry in outcomes] == [
        "failed" if index == failed_scene * 4 + failed_type else "succeeded" for index in range(8)
    ]
    failed = outcomes[failed_scene * 4 + failed_type]
    assert failed["chapter_number"] == 1
    assert failed["error_type"] == ("LLMServiceError" if failure == "model_error" else "ValueError")
    assert failed["error"]
    assert failed["error"] in result["last_error"]
    expected_calls = list(range(8))
    if failure == "invalid_json":
        expected_calls.insert(failed_scene * 4 + failed_type, failed_scene * 4 + failed_type)
    assert calls == expected_calls
    assert manager.get_latest_version("extracted_entities", "chapter_1") == 1
    assert manager.get_latest_version("extracted_relationships", "chapter_1") == 1


async def test_valid_empty_has_complete_ordered_outcomes(extraction_state: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.langgraph.subgraphs.scene_extraction import create_scene_extraction_subgraph

    calls = 0

    async def completion(**arguments: Any) -> tuple[str, None]:
        nonlocal calls
        response = EMPTY_RESPONSES[calls % 4]
        calls += 1
        return json.dumps(response), None

    monkeypatch.setattr(get_services().language_model, "async_call_llm", completion)
    result = await create_scene_extraction_subgraph().ainvoke(extraction_state)
    assert result["extraction_status"] == "complete"
    assert result["extraction_policy"] == "fail_closed"
    assert result["has_fatal_error"] is False
    assert result["current_node"] == "consolidate_extraction"
    assert result["extraction_outcomes"] == [
        {"chapter_number": 1, "scene_index": scene, "extraction_type": kind,
         "status": "succeeded", "item_count": 0, "error_type": "", "error": ""}
        for scene in range(2) for kind in EXTRACTION_TYPES
    ]
    assert calls == 8
    manager = ContentManager(extraction_state["project_dir"])
    assert manager.load_json(result["extracted_entities_ref"]) == {"characters": [], "world_items": []}
    assert manager.load_json(result["extracted_relationships_ref"]) == []


@pytest.mark.parametrize("input_kind", ["missing_ref", "empty_list", "blank_scene", "missing_scene"])
async def test_missing_scene_input_is_not_valid_empty(
    extraction_state: Any, monkeypatch: pytest.MonkeyPatch, input_kind: str,
) -> None:
    from core.langgraph.nodes import scene_extraction

    manager = ContentManager(extraction_state["project_dir"])
    if input_kind == "missing_ref":
        extraction_state["scene_drafts_ref"] = None
    else:
        scenes = {"empty_list": [], "blank_scene": ["   ", "Elara leaves."], "missing_scene": ["Elara enters."]}
        extraction_state["scene_drafts_ref"] = manager.save_list_of_texts(scenes[input_kind], "scenes", "chapter_1", 2)

    async def completion(**arguments: Any) -> tuple[str, None]:
        return json.dumps({"character_updates": {}, "world_updates": {"Location": {}, "Event": {}}, "kg_triples": []}), None

    monkeypatch.setattr(get_services().language_model, "async_call_llm", completion)
    result = await scene_extraction.extract_from_scenes(extraction_state)
    assert result["has_fatal_error"] is True
    assert result["extraction_status"] == "failed"
    assert result["extracted_entities_ref"] is None
    assert result["extracted_relationships_ref"] is None
    assert manager.get_latest_version("extracted_entities", "chapter_1") == 0


@pytest.mark.parametrize("mode", ["success", "late_failure", "all_failure", "json_recovery"])
async def test_provider_boundary_preserves_nonempty_completeness(
    extraction_state: Any, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    import config
    from core.http_client_service import CompletionHTTPClient
    from core.langgraph.subgraphs.scene_extraction import create_scene_extraction_subgraph

    monkeypatch.setitem(vars(config), "settings", config.settings.model_copy(update={"ENABLE_ENTITY_VALIDATION": False}))
    extraction_state["extraction_model"] = "synthetic-primary"
    calls: list[str] = []
    slot = 0
    invalid_sent = False

    async def completion(self: Any, model: str, messages: Any, temperature: float, max_tokens: int, **arguments: Any) -> dict[str, Any]:
        nonlocal slot, invalid_sent
        calls.append(model)
        if mode == "all_failure" or (mode == "late_failure" and slot == 7):
            raise RuntimeError(f"synthetic unavailable {model}")
        if mode == "json_recovery" and slot == 7 and not invalid_sent:
            invalid_sent = True
            return {"choices": [{"message": {"content": "not JSON"}}]}
        scene = slot // 4
        responses: tuple[dict[str, Any], ...] = (
            {"character_updates": {"Elara": {"description": "A scout" if scene == 0 else "A curious scout", "traits": [], "status": "active", "relationships": {}}}},
            {"world_updates": {"Location": {"Library": {"description": "Old", "category": "Structure", "goals": [], "rules": [], "key_elements": []}}}},
            {"world_updates": {"Event": {("Arrival" if scene == 0 else "Departure"): {"description": "Movement", "category": "Travel", "goals": [], "rules": [], "key_elements": []}}}},
            {"kg_triples": [{"subject": "Elara", "predicate": "LOCATED_AT", "object_entity": "Library", "description": "Inside"}]},
        )
        response = responses[slot % 4]
        slot += 1
        return {"choices": [{"message": {"content": json.dumps(response)}}]}

    monkeypatch.setattr(CompletionHTTPClient, "get_completion", completion)
    result = await create_scene_extraction_subgraph().ainvoke(extraction_state)
    manager = ContentManager(extraction_state["project_dir"])
    if mode in ("late_failure", "all_failure"):
        assert result["extraction_status"] == "failed"
        assert result["has_fatal_error"] is True
        assert result["current_node"] == "extract_from_scenes"
        assert result["extracted_entities_ref"] is None
        assert result["extracted_relationships_ref"] is None
        assert manager.get_latest_version("extracted_entities", "chapter_1") == 0
        expected_counts = [0] * 8 if mode == "all_failure" else [1] * 7 + [0]
        assert [outcome["item_count"] for outcome in result["extraction_outcomes"]] == expected_counts
        failures = [outcome for outcome in result["extraction_outcomes"] if outcome["status"] == "failed"]
        assert len(failures) == (8 if mode == "all_failure" else 1)
        for failure in failures:
            assert failure["error_type"] == "LLMServiceError"
            assert "synthetic-primary" in failure["error"]
            assert "RuntimeError" in failure["error"]
            assert "synthetic unavailable" not in failure["error"]
            assert "fallback_error" in failure["error"]
        import config

        assert calls == (["synthetic-primary", config.MEDIUM_MODEL] * 8 if mode == "all_failure" else ["synthetic-primary"] * 8 + [config.MEDIUM_MODEL])
        return

    assert result["extraction_status"] == "complete"
    assert [outcome["item_count"] for outcome in result["extraction_outcomes"]] == [1] * 8
    assert calls == ["synthetic-primary"] * (9 if mode == "json_recovery" else 8)
    assert manager.load_json(result["extracted_entities_ref"]) == {
        "characters": [{"name": "Elara", "type": "Character", "description": "A curious scout", "first_appearance_chapter": 1, "scene_index": 1,
                        "attributes": {"traits": [], "status": "active", "relationships": {}}}],
        "world_items": [
            {"name": "Library", "type": "Location", "description": "Old", "first_appearance_chapter": 1, "scene_index": 0,
             "attributes": {"category": "Structure", "goals": [], "rules": [], "key_elements": []}},
            {"name": "Arrival", "type": "Event", "description": "Movement", "first_appearance_chapter": 1, "scene_index": 0,
             "attributes": {"category": "Travel", "goals": [], "rules": [], "key_elements": []}},
            {"name": "Departure", "type": "Event", "description": "Movement", "first_appearance_chapter": 1, "scene_index": 1,
             "attributes": {"category": "Travel", "goals": [], "rules": [], "key_elements": []}},
        ],
    }
    assert manager.load_json(result["extracted_relationships_ref"]) == [
        {"source_name": "Elara", "target_name": "Library", "relationship_type": "LOCATED_AT", "description": "Inside", "chapter": 1, "scene_index": 0, "confidence": 0.8},
    ]
