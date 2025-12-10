import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.langgraph.initialization.chapter_outline_node import (
    _build_character_summary,
    _determine_act_for_chapter,
    _generate_single_chapter_outline,
    _parse_chapter_outline,
    generate_chapter_outline,
)
from core.langgraph.state import create_initial_state


@pytest.fixture
def base_state():
    """Create a base state for testing."""
    return create_initial_state(
        project_id="test-project",
        title="Test Novel",
        genre="Fantasy",
        theme="Adventure",
        setting="Medieval world",
        target_word_count=80000,
        total_chapters=20,
        project_dir="/tmp/test-project",
        protagonist_name="Hero",
    )


@pytest.fixture
def mock_content_manager():
    """Create a mock ContentManager."""
    with patch(
        "core.langgraph.initialization.chapter_outline_node.ContentManager"
    ) as mock:
        instance = MagicMock()
        instance.get_latest_version.return_value = 0
        instance.save_json.return_value = {
            "path": "mock/path/chapter_outlines.json",
            "size_bytes": 1024,
            "version": 1,
        }
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    with patch(
        "core.langgraph.initialization.chapter_outline_node.llm_service"
    ) as mock:
        mock.async_call_llm = AsyncMock(
            return_value=(
                json.dumps({
                    "scene_description": "Opening scene in the castle",
                    "key_beats": ["Hero awakens", "Receives mission", "Departs castle"],
                    "plot_point": "The quest begins",
                }),
                {"prompt_tokens": 100, "completion_tokens": 50},
            )
        )
        yield mock


@pytest.fixture
def mock_get_functions():
    """Mock the content getter functions."""
    with patch(
        "core.langgraph.initialization.chapter_outline_node.get_chapter_outlines"
    ) as mock_outlines, patch(
        "core.langgraph.initialization.chapter_outline_node.get_global_outline"
    ) as mock_global, patch(
        "core.langgraph.initialization.chapter_outline_node.get_act_outlines"
    ) as mock_acts, patch(
        "core.langgraph.initialization.chapter_outline_node.get_character_sheets"
    ) as mock_chars, patch(
        "core.langgraph.initialization.chapter_outline_node.get_previous_summaries"
    ) as mock_summaries:

        mock_outlines.return_value = {}
        mock_global.return_value = {"act_count": 3, "raw_text": "Global outline text"}
        mock_acts.return_value = {
            1: {"raw_text": "Act 1 outline"},
            2: {"raw_text": "Act 2 outline"},
            3: {"raw_text": "Act 3 outline"},
        }
        mock_chars.return_value = {
            "Hero": {
                "description": "A brave warrior",
                "is_protagonist": True,
            },
            "Sidekick": {
                "description": "Loyal companion",
                "is_protagonist": False,
            },
        }
        mock_summaries.return_value = ["Summary 1", "Summary 2"]

        yield {
            "outlines": mock_outlines,
            "global": mock_global,
            "acts": mock_acts,
            "chars": mock_chars,
            "summaries": mock_summaries,
        }


@pytest.mark.asyncio
async def test_generate_chapter_outline_success(
    base_state, mock_content_manager, mock_llm_service, mock_get_functions
):
    """Verify successful chapter outline generation."""
    state = {**base_state, "current_chapter": 1}

    result = await generate_chapter_outline(state)

    assert result["initialization_step"] == "chapter_outline_1_complete"
    assert result["current_node"] == "chapter_outline"
    assert result["last_error"] is None
    assert "chapter_outlines_ref" in result
    assert result["chapter_outlines_ref"]["size_bytes"] == 1024


@pytest.mark.asyncio
async def test_generate_chapter_outline_already_exists(
    base_state, mock_content_manager, mock_get_functions
):
    """Verify behavior when outline already exists."""
    mock_get_functions["outlines"].return_value = {
        1: {"scene_description": "Existing outline"}
    }

    state = {**base_state, "current_chapter": 1}

    result = await generate_chapter_outline(state)

    assert result["initialization_step"] == "chapter_outline_1_exists"
    assert result["current_node"] == "chapter_outline"


@pytest.mark.asyncio
async def test_generate_chapter_outline_missing_global_outline(
    base_state, mock_content_manager, mock_llm_service, mock_get_functions
):
    """Verify warning when global outline is missing."""
    mock_get_functions["global"].return_value = None

    state = {**base_state, "current_chapter": 1}

    result = await generate_chapter_outline(state)

    assert result["initialization_step"] == "chapter_outline_1_complete"
    assert result["current_node"] == "chapter_outline"


@pytest.mark.asyncio
async def test_generate_chapter_outline_missing_act_outlines(
    base_state, mock_content_manager, mock_llm_service, mock_get_functions
):
    """Verify warning when act outlines are missing."""
    mock_get_functions["acts"].return_value = {}

    state = {**base_state, "current_chapter": 1}

    result = await generate_chapter_outline(state)

    assert result["initialization_step"] == "chapter_outline_1_complete"
    assert result["current_node"] == "chapter_outline"


@pytest.mark.asyncio
async def test_generate_chapter_outline_generation_failure(
    base_state, mock_content_manager, mock_llm_service, mock_get_functions
):
    """Verify handling when generation returns None."""
    mock_llm_service.async_call_llm = AsyncMock(return_value=("", {}))

    state = {**base_state, "current_chapter": 1}

    result = await generate_chapter_outline(state)

    assert result["initialization_step"] == "chapter_outline_1_failed"
    assert result["current_node"] == "chapter_outline"
    assert "Failed to generate outline" in result["last_error"]


@pytest.mark.asyncio
async def test_generate_single_chapter_outline_success(
    base_state, mock_content_manager, mock_llm_service, mock_get_functions
):
    """Verify _generate_single_chapter_outline returns valid outline."""
    state = {**base_state}

    result = await _generate_single_chapter_outline(state, 1, 1)

    assert result is not None
    assert result["chapter_number"] == 1
    assert result["act_number"] == 1
    assert "scene_description" in result
    assert "key_beats" in result
    assert "plot_point" in result
    assert "raw_text" in result


@pytest.mark.asyncio
async def test_generate_single_chapter_outline_empty_response(
    base_state, mock_content_manager, mock_llm_service, mock_get_functions
):
    """Verify handling of empty LLM response."""
    mock_llm_service.async_call_llm = AsyncMock(return_value=("", {}))

    state = {**base_state}

    result = await _generate_single_chapter_outline(state, 1, 1)

    assert result is None


@pytest.mark.asyncio
async def test_generate_single_chapter_outline_exception(
    base_state, mock_content_manager, mock_llm_service, mock_get_functions
):
    """Verify exception handling during generation."""
    mock_llm_service.async_call_llm = AsyncMock(
        side_effect=Exception("LLM error")
    )

    state = {**base_state}

    result = await _generate_single_chapter_outline(state, 1, 1)

    assert result is None


def test_determine_act_for_chapter_three_act():
    """Verify act determination for 3-act structure."""
    with patch(
        "core.langgraph.initialization.chapter_outline_node.ContentManager"
    ) as mock_cm:
        instance = MagicMock()
        mock_cm.return_value = instance

        with patch(
            "core.langgraph.initialization.chapter_outline_node.get_global_outline"
        ) as mock_get:
            mock_get.return_value = {"act_count": 3}

            state = {"total_chapters": 21, "project_dir": "/tmp"}

            assert _determine_act_for_chapter(state, 1) == 1
            assert _determine_act_for_chapter(state, 7) == 1
            assert _determine_act_for_chapter(state, 8) == 2
            assert _determine_act_for_chapter(state, 14) == 2
            assert _determine_act_for_chapter(state, 15) == 3
            assert _determine_act_for_chapter(state, 21) == 3


def test_determine_act_for_chapter_five_act():
    """Verify act determination for 5-act structure."""
    with patch(
        "core.langgraph.initialization.chapter_outline_node.ContentManager"
    ) as mock_cm:
        instance = MagicMock()
        mock_cm.return_value = instance

        with patch(
            "core.langgraph.initialization.chapter_outline_node.get_global_outline"
        ) as mock_get:
            mock_get.return_value = {"act_count": 5}

            state = {"total_chapters": 20, "project_dir": "/tmp"}

            assert _determine_act_for_chapter(state, 1) == 1
            assert _determine_act_for_chapter(state, 4) == 1
            assert _determine_act_for_chapter(state, 5) == 2
            assert _determine_act_for_chapter(state, 8) == 2
            assert _determine_act_for_chapter(state, 9) == 3
            assert _determine_act_for_chapter(state, 12) == 3
            assert _determine_act_for_chapter(state, 13) == 4
            assert _determine_act_for_chapter(state, 16) == 4
            assert _determine_act_for_chapter(state, 17) == 5
            assert _determine_act_for_chapter(state, 20) == 5


def test_determine_act_for_chapter_boundary():
    """Verify act determination doesn't exceed act count."""
    with patch(
        "core.langgraph.initialization.chapter_outline_node.ContentManager"
    ) as mock_cm:
        instance = MagicMock()
        mock_cm.return_value = instance

        with patch(
            "core.langgraph.initialization.chapter_outline_node.get_global_outline"
        ) as mock_get:
            mock_get.return_value = {"act_count": 3}

            state = {"total_chapters": 20, "project_dir": "/tmp"}

            assert _determine_act_for_chapter(state, 99) == 3


def test_build_character_summary_with_characters():
    """Verify character summary with multiple characters."""
    character_sheets = {
        "Hero": {"is_protagonist": True},
        "Villain": {"is_protagonist": False},
        "Sidekick": {"is_protagonist": False},
    }

    result = _build_character_summary(character_sheets)

    assert "Hero" in result
    assert "Protagonist" in result
    assert "Villain" in result or "Sidekick" in result
    assert "Character" in result


def test_build_character_summary_empty():
    """Verify character summary with no characters."""
    result = _build_character_summary({})

    assert result == "No characters defined."


def test_build_character_summary_max_five():
    """Verify character summary limits to 5 characters."""
    character_sheets = {
        f"Character{i}": {"is_protagonist": i == 0}
        for i in range(10)
    }

    result = _build_character_summary(character_sheets)

    lines = [line for line in result.split("\n") if line.strip()]
    assert len(lines) <= 5


def test_parse_chapter_outline_valid_json():
    """Verify parsing of valid JSON response."""
    response = json.dumps({
        "scene_description": "A dark and stormy night",
        "key_beats": ["Thunder strikes", "Door opens", "Stranger enters"],
        "plot_point": "The visitor arrives",
    })

    result = _parse_chapter_outline(response, 5, 2)

    assert result["chapter_number"] == 5
    assert result["act_number"] == 2
    assert result["scene_description"] == "A dark and stormy night"
    assert len(result["key_beats"]) == 3
    assert result["plot_point"] == "The visitor arrives"
    assert result["raw_text"] == response


def test_parse_chapter_outline_json_with_markdown():
    """Verify parsing of JSON wrapped in markdown code blocks."""
    response = """```json
{
    "scene_description": "Test scene",
    "key_beats": ["Beat 1"],
    "plot_point": "Test plot"
}
```"""

    result = _parse_chapter_outline(response, 1, 1)

    assert result["scene_description"] == "Test scene"
    assert result["key_beats"] == ["Beat 1"]
    assert result["plot_point"] == "Test plot"


def test_parse_chapter_outline_invalid_json_fallback():
    """Verify fallback parsing when JSON is invalid."""
    response = """Scene: The hero enters the castle

Beats:
- Meets the king
- Receives quest
- Departs on journey

Plot Point: The adventure begins"""

    result = _parse_chapter_outline(response, 3, 1)

    assert result["chapter_number"] == 3
    assert result["act_number"] == 1
    assert result["raw_text"] == response
    assert len(result["scene_description"]) > 0


def test_parse_chapter_outline_empty_response():
    """Verify parsing handles empty response."""
    response = ""

    result = _parse_chapter_outline(response, 1, 1)

    assert result["chapter_number"] == 1
    assert result["act_number"] == 1
    assert result["scene_description"] == ""
    assert result["key_beats"] == []


def test_parse_chapter_outline_limits_beats():
    """Verify parsing limits key beats to 10."""
    response = json.dumps({
        "scene_description": "Test",
        "key_beats": [f"Beat {i}" for i in range(20)],
        "plot_point": "Test",
    })

    result = _parse_chapter_outline(response, 1, 1)

    assert len(result["key_beats"]) == 10


def test_parse_chapter_outline_text_with_beats():
    """Verify text parsing extracts beats correctly."""
    response = """Scene Description: Opening scene

Key Beats:
- Hero wakes up
- Discovers message
- Makes decision

Plot Point: The journey begins"""

    result = _parse_chapter_outline(response, 1, 1)

    assert len(result["key_beats"]) >= 1


def test_parse_chapter_outline_fallback_uses_full_text():
    """Verify fallback uses full text when no structure found."""
    response = "Just a plain text outline with no structure."

    result = _parse_chapter_outline(response, 1, 1)

    assert result["scene_description"] == response[:500]
    assert result["chapter_number"] == 1
    assert result["act_number"] == 1


@pytest.mark.asyncio
async def test_generate_chapter_outline_different_chapters(
    base_state, mock_content_manager, mock_llm_service, mock_get_functions
):
    """Verify generation works for different chapter numbers."""
    for chapter_num in [1, 5, 10, 15, 20]:
        state = {**base_state, "current_chapter": chapter_num}

        result = await generate_chapter_outline(state)

        assert result["initialization_step"] == f"chapter_outline_{chapter_num}_complete"
        assert result["current_node"] == "chapter_outline"
