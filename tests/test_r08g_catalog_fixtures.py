"""Selected synthetic catalogs for scene tests; never infer names from responses."""

from pathlib import Path

from core.langgraph.content_manager import ContentManager
from core.langgraph.initialization.catalog import materialize_entities, select_catalog
from core.langgraph.initialization.snapshot import select_inputs
from core.langgraph.state import NarrativeState
from models.kg_models import WorldItem
from tests.test_staged_initialization import example_state


def catalog_state(directory: Path, characters: tuple[str, ...] = ("Elara",), locations: tuple[str, ...] = ("Library",), events: tuple[str, ...] = (), existing: NarrativeState | None = None) -> NarrativeState:
    state: NarrativeState = {**(existing or {}), **example_state(directory)}
    manager = ContentManager(str(directory))
    reference = state["character_sheets_ref"]
    assert reference is not None
    sheet = manager.load_json_strict(reference)["Ada"]
    state["character_sheets_ref"] = manager.save_json({name: {**sheet, "name": name} for name in characters}, "character_sheets", "scene_fixture", 1)
    if events:
        chapter_reference = state["chapter_outlines_ref"]
        assert chapter_reference is not None
        chapters = manager.load_json_strict(chapter_reference)
        chapters["1"]["key_beats"] = list(events)
        chapters["1"]["version"] = 1
        state["chapter_outlines_ref"] = manager.save_json(chapters, "chapter_outlines", "scene_fixture", 1)
    catalog = materialize_entities(select_inputs(state), [WorldItem(name=name, category="location", description="Selected synthetic location", id=f"location-{index}") for index, name in enumerate(locations)], ())
    state["initialization_catalog_ref"] = manager.save_json(catalog.model_dump(mode="json"), "initialization_catalog", catalog.inputs.identity, 2)
    select_catalog(state)
    return state


def test_catalog_fixture_selects_exact_named_identities(tmp_path: Path) -> None:
    state = catalog_state(tmp_path, events=("Arrival", "Departure"))
    candidates = select_catalog(state).candidates("Character", "Location", "Event")
    assert {row["name"] for row in candidates} >= {"Elara", "Library", "Arrival", "Departure"}
    assert all(row["id"] for row in candidates)
