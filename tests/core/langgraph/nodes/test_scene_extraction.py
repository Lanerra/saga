# tests/core/langgraph/nodes/test_scene_extraction.py
from typing import Any
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_extract_from_scene_returns_entities(tmp_path: Any) -> None:
    from core.langgraph.nodes.scene_extraction import extract_from_scene

    scene_text = "Elara walked into the Sunken Library and found the ancient map."
    scene_index = 0
    chapter_number = 1

    mock_char_response = {
        "character_updates": {
            "Elara": {
                "description": "A brave explorer",
                "traits": ["brave"],
                "status": "active",
                "relationships": {},
            }
        }
    }
    mock_loc_response = {
        "world_updates": {
            "Location": {
                "Sunken Library": {
                    "description": "An ancient library",
                    "category": "Structure",
                    "goals": [],
                    "rules": [],
                    "key_elements": ["books"],
                }
            }
        }
    }
    mock_event_response: dict[str, Any] = {"world_updates": {"Event": {}}}
    mock_rel_response: dict[str, Any] = {"kg_triples": []}

    async def mock_llm_json(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], None]:
        prompt = kwargs.get("prompt", "")
        if "character extraction" in prompt.lower():
            return mock_char_response, None
        if "location extraction" in prompt.lower():
            return mock_loc_response, None
        if "event extraction" in prompt.lower():
            return mock_event_response, None
        return mock_rel_response, None

    with patch(
        "core.llm_interface_refactored.llm_service.async_call_llm_json_object",
        side_effect=mock_llm_json,
    ):
        result = await extract_from_scene(
            scene_text=scene_text,
            scene_index=scene_index,
            chapter_number=chapter_number,
            novel_title="Test Novel",
            novel_genre="Fantasy",
            protagonist_name="Elara",
            model_name="test-model",
        )

    assert "characters" in result
    assert "world_items" in result
    assert "relationships" in result
    assert len(result["characters"]) == 1
    assert result["characters"][0]["name"] == "Elara"


def test_consolidate_scene_extractions_deduplicates_by_name() -> None:
    from core.langgraph.nodes.scene_extraction import consolidate_scene_extractions

    scene_results = [
        {
            "characters": [
                {"name": "Elara", "type": "Character", "description": "A hero", "attributes": {}},
            ],
            "world_items": [
                {"name": "Library", "type": "Location", "description": "Old", "attributes": {}},
            ],
            "relationships": [],
        },
        {
            "characters": [
                {"name": "Elara", "type": "Character", "description": "A brave hero", "attributes": {}},
                {"name": "Marcus", "type": "Character", "description": "A friend", "attributes": {}},
            ],
            "world_items": [
                {"name": "Library", "type": "Location", "description": "Ancient", "attributes": {}},
            ],
            "relationships": [
                {"source_name": "Elara", "target_name": "Marcus", "relationship_type": "KNOWS"},
            ],
        },
    ]

    result = consolidate_scene_extractions(scene_results)

    character_names = [c["name"] for c in result["characters"]]
    assert character_names.count("Elara") == 1
    assert "Marcus" in character_names

    location_names = [w["name"] for w in result["world_items"]]
    assert location_names.count("Library") == 1

    assert len(result["relationships"]) == 1


@pytest.mark.asyncio
async def test_extract_from_scenes_node_processes_all_scenes(tmp_path: Any) -> None:
    from core.langgraph.content_manager import ContentManager
    from core.langgraph.nodes.scene_extraction import extract_from_scenes
    from core.langgraph.state import create_initial_state

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
    scenes = [
        "Scene 1: Elara enters the library.",
        "Scene 2: She meets Marcus at the tower.",
    ]
    state["scene_drafts_ref"] = content_manager.save_list_of_texts(scenes, "scenes", "chapter_1", 1)
    state["current_chapter"] = 1

    async def mock_llm(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], None]:
        return {"character_updates": {}, "world_updates": {"Location": {}, "Event": {}}, "kg_triples": []}, None

    with patch(
        "core.llm_interface_refactored.llm_service.async_call_llm_json_object",
        side_effect=mock_llm,
    ):
        result = await extract_from_scenes(state)

    assert "extracted_entities" in result
    assert "extracted_relationships" in result
    assert result["current_node"] == "extract_from_scenes"
