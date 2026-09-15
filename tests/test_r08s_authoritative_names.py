"""Selected author identity outranks lexical heuristics; unselected names remain closed."""

from pathlib import Path

import pytest

from core.langgraph.content_manager import ContentManager
from tests.test_r02a_named_candidates import run_extraction
from tests.test_staged_initialization import example_state, with_catalog


@pytest.mark.parametrize("name", ["Power", "e.e. cummings"])
async def test_authorized_name_survives_real_parsers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    state = example_state(tmp_path)
    manager = ContentManager(str(tmp_path))
    reference = state["character_sheets_ref"]
    assert reference is not None
    sheets = manager.load_json_strict(reference)
    sheets[name] = {**sheets["Ada"], "name": name, "is_protagonist": False}
    state["character_sheets_ref"] = manager.save_json(sheets, "character_sheets", "author", 1)
    state = with_catalog(state)
    state["scene_drafts_ref"] = manager.save_json([f"Ada trusts {name}. Neighbors watch."], "scene_drafts", "chapter_1", 1)
    result, requests = await run_extraction(monkeypatch, state, [
        {"character_updates": {name: {"description": "An author-named character", "traits": [], "status": "Active", "relationships": {}}}},
        {"kg_triples": [{"subject": "Ada", "predicate": "TRUSTS", "object_entity": name, "description": "Synthetic assertion"}]},
    ])
    assert result["extraction_status"] == "complete", result.get("last_error")
    # No selected Location/Event occurs in this scene: those producers send nothing.
    assert [request["response_format"]["json_schema"]["name"] for request in requests] == [
        "extract_scene_characters", "extract_scene_relationships",
    ]
    rows = manager.load_json_strict(result["extracted_relationships_ref"])
    assert rows[0]["target_name"] == name
    assert all(row["source_name"] != "Neighbors" and row["target_name"] != "Neighbors" for row in rows)
