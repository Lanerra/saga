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
