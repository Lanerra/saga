"""Strict catalog and selected-context mechanics, using synthetic transports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.langgraph.content_manager import ContentManager
from core.langgraph.initialization.catalog import EntityCatalog, ProducerEvidence, materialize_entities, materialize_initialization_catalog
from core.langgraph.initialization.chapter_outline_node import _generate_single_chapter_outline
from core.langgraph.initialization.commit_init_node import _parse_world_items_extraction
from core.langgraph.initialization.snapshot import select_inputs
from core.service_context import get_services
from tests.test_staged_initialization import example_state
from tests.test_r05_initialization_contracts import chapter_response


@pytest.mark.parametrize("case", ["missing_character", "changed_character", "missing_event", "wrong_scene"])
def test_catalog_rejects_materialization_drift(tmp_path: Path, case: str) -> None:
    catalog = materialize_entities(select_inputs(example_state(tmp_path)), [], ())
    data = catalog.model_dump(mode="json")
    if case.startswith("missing"):
        label = "Character" if case == "missing_character" else "Event"
        selected = next(entity for entity in data["entities"] if entity["label"] == label)
        data["entities"].remove(selected)
    else:
        label = "Character" if case == "changed_character" else "Scene"
        selected = next(entity for entity in data["entities"] if entity["label"] == label)
        payload = json.loads(selected["payload"])
        payload["motivations" if case == "changed_character" else "setting"] = "Unselected replacement"
        selected["payload"] = json.dumps(payload)
    with pytest.raises(ValueError):
        EntityCatalog.model_validate_json(json.dumps(data))


async def test_chapter_prompt_uses_all_selected_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = example_state(tmp_path)
    manager = ContentManager(str(tmp_path))
    assert state["character_sheets_ref"] is not None
    sheets = manager.load_json_strict(state["character_sheets_ref"])
    for name in ["Bea", "Cora", "Dara"]:
        sheets[name] = {**sheets["Ada"], "name": name, "motivations": f"{name} selected motivation", "is_protagonist": False}
    state["character_sheets_ref"] = manager.save_json(sheets, "character_sheets", "all", version=2)
    assert state["global_outline_ref"] is not None
    outline = manager.load_json_strict(state["global_outline_ref"])
    outline["thematic_progression"] = "Selected thematic edit not in raw text"
    state["global_outline_ref"] = manager.save_json(outline, "global_outline", "all", version=2)
    prompts: list[str] = []

    async def transport(**arguments: Any) -> tuple[str, dict[str, Any]]:
        prompts.append(arguments["prompt"])
        return json.dumps(chapter_response()), {}

    monkeypatch.setattr(get_services().language_model, "async_call_llm", transport)
    result = await _generate_single_chapter_outline(state, 1, 1)
    assert result is not None
    assert len(prompts) == 1
    assert "Dara selected motivation" in prompts[0]
    assert outline["thematic_progression"] in prompts[0]


async def test_world_materialization_sees_character_and_chapter_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = example_state(tmp_path)
    state["outline_relationships_ref"] = None
    manager = ContentManager(str(tmp_path))
    assert state["chapter_outlines_ref"] is not None
    chapters = manager.load_json_strict(state["chapter_outlines_ref"])
    chapters["1"]["scene_description"] = "Ada visits Unique Chapter Observatory."
    state["chapter_outlines_ref"] = manager.save_json(chapters, "chapter_outlines", "selected", version=0)
    assert state["character_sheets_ref"] is not None
    sheets = manager.load_json_strict(state["character_sheets_ref"])
    sheets["Ada"]["background"] = "Keeps Unique Character Compass."
    state["character_sheets_ref"] = manager.save_json(sheets, "character_sheets", "selected", version=1)
    prompts: list[str] = []

    async def transport(**arguments: Any) -> tuple[str, dict[str, Any]]:
        prompts.append(arguments["prompt"])
        return "[]", {}

    monkeypatch.setattr(get_services().language_model, "async_call_llm", transport)
    await materialize_initialization_catalog(state)
    assert len(prompts) == 1
    assert "Unique Chapter Observatory" in prompts[0]
    assert "Unique Character Compass" in prompts[0]


def test_catalog_chapter_summary_preserves_produced_scene_description(tmp_path: Path) -> None:
    inputs = select_inputs(example_state(tmp_path))
    catalog = materialize_entities(inputs, [], ())
    chapter = next(json.loads(entity.payload) for entity in catalog.entities if entity.label == "Chapter")
    assert chapter["summary"] == inputs.chapters[0].scene_description


@pytest.mark.parametrize("response", [
    '[{"name":"discarded","name":"Compass","category":"object","description":"Brass"}]',
    'text [{"name":"Compass","category":"object","description":"Brass"}]',
    '[{"name":" Compass ","category":"object","description":"Brass"}]',
])
def test_world_output_rejects_repaired_identity(response: str) -> None:
    with pytest.raises(ValueError):
        _parse_world_items_extraction(response)


def test_world_output_preserves_description_bytes() -> None:
    description = " Brass" + chr(10)
    items = _parse_world_items_extraction(json.dumps([dict(name="Compass", category="object", description=description)]))
    assert items[0].description == description


@pytest.mark.parametrize("case", ["missing", "changed"])
def test_catalog_preserves_world_producer_evidence(tmp_path: Path, case: str) -> None:
    response = '[{"name":"Compass","category":"object","description":" Brass"}]'
    evidence = ProducerEvidence(model="synthetic", template="knowledge_agent/extract_world_items_lines.j2", prompt_checksum="test", response=response)
    catalog = materialize_entities(select_inputs(example_state(tmp_path)), _parse_world_items_extraction(response), (evidence,))
    data = catalog.model_dump(mode="json")
    item = next(entity for entity in data["entities"] if entity["label"] == "Item")
    if case == "missing":
        data["entities"].remove(item)
    else:
        payload = json.loads(item["payload"])
        payload["description"] = "Unselected replacement"
        item["payload"] = json.dumps(payload)
    with pytest.raises(ValueError, match="world"):
        EntityCatalog.model_validate_json(json.dumps(data))
