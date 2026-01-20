# tests/core/langgraph/nodes/test_scene_extraction.py
import json
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel


def _assert_no_pydantic_models(value: Any) -> None:
    if isinstance(value, BaseModel):
        raise AssertionError(f"Found Pydantic model in state: {type(value)}")
    if isinstance(value, dict):
        for nested in value.values():
            _assert_no_pydantic_models(nested)
        return
    if isinstance(value, list):
        for nested in value:
            _assert_no_pydantic_models(nested)
        return
    if isinstance(value, tuple):
        for nested in value:
            _assert_no_pydantic_models(nested)
        return


def _assert_json_serializable(value: Any) -> None:
    json.dumps(value)


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

    # Data is now externalized immediately, so only refs should be in result
    assert "extracted_entities_ref" in result
    assert "extracted_relationships_ref" in result
    assert result["current_node"] == "extract_from_scenes"
    _assert_no_pydantic_models(result)
    _assert_json_serializable(result)


@pytest.mark.asyncio
async def test_extract_from_scenes_no_scenes_returns_empty_serializable_state(tmp_path: Any) -> None:
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
    state["current_chapter"] = 1

    result = await extract_from_scenes(state)

    # When no scenes, refs should still be created (pointing to empty files)
    assert "extracted_entities_ref" in result
    assert "extracted_relationships_ref" in result
    _assert_no_pydantic_models(result)
    _assert_json_serializable(result)


@pytest.mark.asyncio
async def test_extract_from_scenes_converts_pydantic_models_to_dicts(tmp_path: Any) -> None:
    from core.langgraph.content_manager import ContentManager
    from core.langgraph.nodes.scene_extraction import extract_from_scenes
    from core.langgraph.state import ExtractedEntity, ExtractedRelationship, create_initial_state

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
    state["scene_drafts_ref"] = content_manager.save_list_of_texts(
        ["Scene 1: Elara enters the library."],
        "scenes",
        "chapter_1",
        1,
    )
    state["current_chapter"] = 1

    extracted_entity = ExtractedEntity(
        name="Elara",
        type="Character",
        description="A brave explorer",
        first_appearance_chapter=1,
        attributes={"traits": ["brave"]},
    )
    extracted_relationship = ExtractedRelationship(
        source_name="Elara",
        target_name="Marcus",
        relationship_type="KNOWS",
        description="They met recently",
        chapter=1,
        confidence=0.8,
    )

    def fake_consolidate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "characters": [extracted_entity],
            "world_items": [],
            "relationships": [extracted_relationship],
        }

    async def fake_extract_from_scene(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"characters": [], "world_items": [], "relationships": []}

    with (
        patch(
            "core.langgraph.nodes.scene_extraction.consolidate_scene_extractions",
            side_effect=fake_consolidate,
        ),
        patch(
            "core.langgraph.nodes.scene_extraction.extract_from_scene",
            side_effect=fake_extract_from_scene,
        ),
        patch(
            "core.llm_interface_refactored.llm_service.async_call_llm_json_object",
            return_value=({"character_updates": {}, "world_updates": {"Location": {}, "Event": {}}, "kg_triples": []}, None),
        ),
    ):
        result = await extract_from_scenes(state)

    _assert_no_pydantic_models(result)
    _assert_json_serializable(result)

    # Data is externalized, so in-memory state should be cleared
    assert "extracted_entities" not in result or result.get("extracted_entities") == {}
    assert "extracted_relationships" not in result or result.get("extracted_relationships") == []
    assert result["extracted_entities_ref"] is not None
    assert result["extracted_relationships_ref"] is not None

    from core.langgraph.content_manager import ContentManager

    content_manager = ContentManager(state["project_dir"])
    extracted_entities = content_manager.load_json(result["extracted_entities_ref"])
    extracted_relationships = content_manager.load_json(result["extracted_relationships_ref"])

    characters = extracted_entities["characters"]
    assert isinstance(characters, list)
    assert characters == [
        {
            "name": "Elara",
            "type": "Character",
            "description": "A brave explorer",
            "first_appearance_chapter": 1,
            "attributes": {"traits": ["brave"]},
        }
    ]

    assert extracted_relationships == [
        {
            "source_name": "Elara",
            "target_name": "Marcus",
            "relationship_type": "KNOWS",
            "description": "They met recently",
            "chapter": 1,
            "confidence": 0.8,
            "source_type": None,
            "target_type": None,
        }
    ]
