"""Selected synthetic mechanics fixtures; no provider or narrative-quality evidence."""

import hashlib
import importlib
from pathlib import Path

import pytest

from core.langgraph.content_manager import ContentManager
from core.langgraph.initialization.catalog import materialize_entities, select_catalog
from core.langgraph.initialization.snapshot import select_inputs
from core.langgraph.state import NarrativeState
from models.kg_models import WorldItem
from tests.test_staged_initialization import example_state, with_catalog


def selected_authority_state(
    directory: Path,
    character_names: tuple[str, ...],
    *,
    world_items: tuple[WorldItem, ...] = (),
    event_names: tuple[str, ...] = (),
    chapters: int = 1,
    initial_state: NarrativeState | None = None,
) -> NarrativeState:
    state: NarrativeState = {**(initial_state or {}), **example_state(directory)}
    manager = ContentManager(str(directory))
    reference = state["character_sheets_ref"]
    assert reference is not None
    sheet = manager.load_json_strict(reference)["Ada"]
    state["character_sheets_ref"] = manager.save_json(
        {name: {**sheet, "name": name, "is_protagonist": index == 0} for index, name in enumerate(character_names)},
        "character_sheets", "selected_author", 1,
    )
    global_reference = state["global_outline_ref"]
    act_reference = state["act_outlines_ref"]
    chapter_reference = state["chapter_outlines_ref"]
    assert global_reference is not None and act_reference is not None and chapter_reference is not None
    global_outline = manager.load_json_strict(global_reference)
    global_outline["total_chapters"] = chapters
    global_outline["acts"][0]["chapters_end"] = chapters
    acts = manager.load_json_strict(act_reference)
    acts["acts"][0]["chapters_in_act"] = chapters
    chapter = manager.load_json_strict(chapter_reference)["1"]
    state["total_chapters"] = chapters
    state["global_outline_ref"] = manager.save_json(global_outline, "global_outline", "selected_author", 1)
    state["act_outlines_ref"] = manager.save_json(acts, "act_outlines", "selected_author", 1)
    state["chapter_outlines_ref"] = manager.save_json(
        {str(number): {**chapter, "chapter_number": number, "key_beats": list(event_names) if event_names else chapter["key_beats"]} for number in range(1, chapters + 1)},
        "chapter_outlines", "selected_author", 0,
    )
    if world_items:
        catalog = materialize_entities(select_inputs(state), list(world_items), ())
        state["initialization_catalog_ref"] = manager.save_json(catalog.model_dump(mode="json"), "initialization_catalog", catalog.inputs.identity, 2)
    else:
        state = with_catalog(state)
    select_catalog(state)
    return state


@pytest.mark.parametrize("module_name", [
    "config", "core.graph_healing_service", "core.langgraph.initialization.catalog",
    "core.langgraph.nodes.scene_extraction", "core.langgraph.nodes.scene_extraction_validation",
    "core.langgraph.nodes.scene_planning_node", "core.langgraph.nodes.scene_generation_node",
    "core.langgraph.nodes.narrative_enrichment_node", "core.http_client_service",
])
def test_source_import_provenance(module_name: str) -> None:
    module = importlib.import_module(module_name)
    relative = "config/__init__.py" if module_name == "config" else module_name.replace(".", "/") + ".py"
    assert module.__file__ is not None
    assert Path(module.__file__).resolve() == Path(__file__).resolve().parents[1] / relative


def test_authority_fixture_materializes_names_from_selected_sources(tmp_path: Path) -> None:
    state = selected_authority_state(tmp_path, ("Father O'Brien", "Neighbors"), event_names=("The King's Return",))
    catalog = select_catalog(state)
    assert [row["name"] for row in catalog.candidates("Character")] == ["Father O'Brien", "Neighbors"]
    assert any(row["name"] == "The King's Return" for row in catalog.candidates("Event"))
    assert [sheet.name for sheet in catalog.inputs.characters] == ["Father O'Brien", "Neighbors"]


def test_frozen_retained_request_response_literal_is_unchanged() -> None:
    import ast

    path = Path(__file__).with_name("test_r02_extraction_identity.py")
    assignment = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "RETAINED" for target in node.targets))
    assert isinstance(assignment.value, ast.Call)
    literal = ast.literal_eval(assignment.value.args[0])
    assert hashlib.sha256(literal.encode()).hexdigest() == "fa8d934469813c60f6ef400439eb7068d84b92b93e333afd3392304e12318e3a"


async def test_enriched_extraction_preserves_every_selected_authority_byte(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_r08s_producer_composition import extract_selected, prepare_enriched_state

    state, _ = await prepare_enriched_state(tmp_path, monkeypatch)
    catalog = select_catalog(state, retained_chapter_outline=True)
    reference = state["initialization_catalog_ref"]
    assert reference is not None
    paths = [tmp_path / artifact.path for artifact in catalog.inputs.artifacts] + [tmp_path / reference["path"]]
    before = {path: path.read_bytes() for path in paths}
    original_state = dict(state)
    result, requests = await extract_selected(state, monkeypatch)
    assert result["extraction_status"] == "complete", result.get("last_error")
    assert len(requests) == 8
    assert state == original_state
    assert {path: path.read_bytes() for path in paths} == before
    assert select_catalog(state, retained_chapter_outline=True) == catalog
