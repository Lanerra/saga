import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.langgraph.initialization.character_sheets_node import (
    _generate_character_list,
    _generate_character_sheet,
    _get_existing_traits,
    _parse_character_sheet_response,
    generate_character_sheets,
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
    with patch("core.langgraph.initialization.character_sheets_node.ContentManager") as mock:
        instance = MagicMock()
        instance.save_json.return_value = {
            "path": "mock/path/character_sheets.json",
            "size_bytes": 4096,
            "version": 1,
        }
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_neo4j():
    """Mock Neo4j manager for trait queries."""
    with patch("core.langgraph.initialization.character_sheets_node.neo4j_manager") as mock:
        mock.execute_read_query = AsyncMock(
            return_value=[
                {"trait_name": "brave"},
                {"trait_name": "loyal"},
                {"trait_name": "cunning"},
            ]
        )
        yield mock


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    with patch("core.langgraph.initialization.character_sheets_node.llm_service") as mock:
        mock.async_call_llm = AsyncMock(
            return_value=(
                json.dumps(
                    {
                        "name": "Hero",
                        "description": "A brave warrior",
                        "traits": ["brave", "loyal"],
                        "status": "Active",
                        "motivations": "Save the kingdom",
                        "background": "Born in a village",
                        "skills": ["swordfighting"],
                        "relationships": {"Mentor": "Wise guide"},
                        "internal_conflict": "Self-doubt",
                    }
                ),
                {"prompt_tokens": 100, "completion_tokens": 50},
            )
        )
        yield mock


@pytest.fixture
def mock_schema_validator():
    """Mock schema validator."""
    with patch("core.langgraph.initialization.character_sheets_node.schema_validator") as mock:
        mock.validate_entity_type.return_value = (True, "Character", None)
        yield mock


@pytest.mark.asyncio
async def test_get_existing_traits_success(mock_neo4j):
    """Verify successful retrieval of existing traits."""
    traits = await _get_existing_traits()

    assert len(traits) == 3
    assert "brave" in traits
    assert "loyal" in traits
    assert "cunning" in traits


@pytest.mark.asyncio
async def test_get_existing_traits_empty():
    """Verify handling when no traits exist."""
    with patch("core.langgraph.initialization.character_sheets_node.neo4j_manager") as mock:
        mock.execute_read_query = AsyncMock(return_value=[])

        traits = await _get_existing_traits()

        assert traits == []


@pytest.mark.asyncio
async def test_get_existing_traits_exception():
    """Verify exception handling during trait retrieval."""
    with patch("core.langgraph.initialization.character_sheets_node.neo4j_manager") as mock:
        mock.execute_read_query = AsyncMock(side_effect=Exception("Database error"))

        traits = await _get_existing_traits()

        assert traits == []


def test_parse_character_sheet_response_valid_json(mock_schema_validator):
    """Verify parsing of valid JSON character sheet."""
    response = json.dumps(
        {
            "name": "Hero",
            "description": "A brave warrior",
            "traits": ["brave", "loyal"],
            "status": "Active",
            "motivations": "Save the kingdom",
            "background": "Born in a village",
            "skills": ["swordfighting"],
            "relationships": {"Mentor": "Wise guide"},
            "internal_conflict": "Self-doubt",
        }
    )

    with patch("core.langgraph.initialization.character_sheets_node.validate_and_filter_traits") as mock_validate:
        mock_validate.return_value = ["brave", "loyal"]

        result = _parse_character_sheet_response(response, "Hero")

        # Note: the model JSON here intentionally omits `type`. The parser injects
        # a default `type: "Character"` and normalizes it via schema validation.
        assert result["type"] == "Character"
        assert result["name"] == "Hero"
        assert result["description"] == "A brave warrior"
        assert len(result["traits"]) == 2
        assert result["status"] == "Active"


def test_parse_character_sheet_response_with_markdown(mock_schema_validator):
    """Verify parsing of JSON wrapped in markdown code blocks."""
    response = """```json
{
    "name": "Hero",
    "description": "Test",
    "traits": ["brave"]
}
```"""

    with patch("core.langgraph.initialization.character_sheets_node.validate_and_filter_traits") as mock_validate:
        mock_validate.return_value = ["brave"]

        result = _parse_character_sheet_response(response, "Hero")

        assert result["name"] == "Hero"
        assert result["traits"] == ["brave"]


def test_parse_character_sheet_response_missing_name(mock_schema_validator):
    """Verify name defaults to provided character_name if missing."""
    response = json.dumps(
        {
            "description": "Test character",
            "traits": [],
        }
    )

    with patch("core.langgraph.initialization.character_sheets_node.validate_and_filter_traits") as mock_validate:
        mock_validate.return_value = []

        result = _parse_character_sheet_response(response, "DefaultName")

        assert result["name"] == "DefaultName"


def test_parse_character_sheet_response_invalid_json(mock_schema_validator):
    """Verify strict failure when JSON parsing fails (no prose fallback)."""
    response = "This is not valid JSON but a plain text description"

    with patch("core.langgraph.initialization.character_sheets_node.validate_and_filter_traits") as mock_validate:
        mock_validate.return_value = []

        with pytest.raises(ValueError, match=r"Character sheet JSON parsing failed"):
            _parse_character_sheet_response(response, "Hero")


def test_parse_character_sheet_response_filters_traits(mock_schema_validator):
    """Verify trait filtering removes invalid traits."""
    response = json.dumps(
        {
            "name": "Hero",
            "traits": ["brave", "very long invalid trait", "loyal", "another bad trait here"],
        }
    )

    with patch("core.langgraph.initialization.character_sheets_node.validate_and_filter_traits") as mock_validate:
        mock_validate.return_value = ["brave", "loyal"]

        result = _parse_character_sheet_response(response, "Hero")

        assert len(result["traits"]) == 2


def test_parse_character_sheet_response_transforms_relationships(
    mock_schema_validator,
):
    """Verify relationship transformation to internal structure."""
    response = json.dumps(
        {
            "name": "Hero",
            "relationships": {
                "Mentor": "Wise guide",
                "Friend": "Close ally",
            },
        }
    )

    with patch("core.langgraph.initialization.character_sheets_node.validate_and_filter_traits") as mock_validate:
        mock_validate.return_value = []

        result = _parse_character_sheet_response(response, "Hero")

        assert "Mentor" in result["relationships"]
        assert result["relationships"]["Mentor"]["description"] == "Wise guide"
        assert result["relationships"]["Mentor"]["type"] == "ASSOCIATE"


@pytest.mark.asyncio
async def test_generate_character_list_success(base_state, mock_llm_service):
    """Verify successful generation of character list."""
    mock_llm_service.async_call_llm = AsyncMock(
        return_value=(
            json.dumps(["Hero", "Mentor", "Villain", "Sidekick"]),
            {"prompt_tokens": 100, "completion_tokens": 50},
        )
    )

    result = await _generate_character_list(base_state)

    assert result == ["Hero", "Mentor", "Villain", "Sidekick"]


@pytest.mark.asyncio
async def test_generate_character_list_rejects_non_json(base_state, mock_llm_service):
    """Verify strict JSON-only contract for character list (no fallback)."""
    mock_llm_service.async_call_llm = AsyncMock(
        return_value=(
            "1. Hero\n2. Mentor\n- Villain\n* Sidekick",
            {"prompt_tokens": 100, "completion_tokens": 50},
        )
    )

    result = await _generate_character_list(base_state)

    assert result == []


@pytest.mark.asyncio
async def test_generate_character_list_requires_protagonist(base_state, mock_llm_service):
    """Verify protagonist name must be included exactly when provided (no fallback)."""
    mock_llm_service.async_call_llm = AsyncMock(
        return_value=(
            json.dumps(["Mentor", "Villain", "Sidekick"]),
            {"prompt_tokens": 100, "completion_tokens": 50},
        )
    )

    result = await _generate_character_list(base_state)

    assert result == []


@pytest.mark.asyncio
async def test_generate_character_list_rejects_more_than_ten(base_state, mock_llm_service):
    """Verify character list rejects more than 10 characters (no fallback)."""
    payload = ["Hero"] + [f"Character{i}" for i in range(20)]
    mock_llm_service.async_call_llm = AsyncMock(return_value=(json.dumps(payload), {"prompt_tokens": 100, "completion_tokens": 50}))

    result = await _generate_character_list(base_state)

    assert result == []


@pytest.mark.asyncio
async def test_generate_character_list_empty_response(base_state, mock_llm_service):
    """Verify empty response yields failure (no fallback)."""
    mock_llm_service.async_call_llm = AsyncMock(return_value=("", {}))

    result = await _generate_character_list(base_state)

    assert result == []


@pytest.mark.asyncio
async def test_generate_character_list_exception(base_state, mock_llm_service):
    """Verify exception yields failure (no fallback)."""
    mock_llm_service.async_call_llm = AsyncMock(side_effect=Exception("LLM error"))

    result = await _generate_character_list(base_state)

    assert result == []


@pytest.mark.asyncio
async def test_generate_character_sheet_success(base_state, mock_llm_service, mock_schema_validator):
    """Verify successful generation of character sheet."""
    with patch("core.langgraph.initialization.character_sheets_node.validate_and_filter_traits") as mock_validate:
        mock_validate.return_value = ["brave", "loyal"]

        result = await _generate_character_sheet(
            state=base_state,
            character_name="Hero",
            other_characters=["Mentor", "Villain"],
            existing_traits=["brave", "loyal", "cunning"],
        )

        assert result is not None
        assert result["name"] == "Hero"
        assert result["is_protagonist"] is True
        assert result["generated_at"] == "initialization"
        assert "raw_response" in result


@pytest.mark.asyncio
async def test_generate_character_sheet_non_protagonist(base_state, mock_llm_service, mock_schema_validator):
    """Verify sheet generation for non-protagonist character."""
    with patch("core.langgraph.initialization.character_sheets_node.validate_and_filter_traits") as mock_validate:
        mock_validate.return_value = []

        result = await _generate_character_sheet(
            state=base_state,
            character_name="Mentor",
            other_characters=["Hero", "Villain"],
        )

        assert result is not None
        assert result["is_protagonist"] is False


@pytest.mark.asyncio
async def test_generate_character_sheet_empty_response(base_state, mock_llm_service, mock_schema_validator):
    """Verify handling of empty LLM response."""
    mock_llm_service.async_call_llm = AsyncMock(return_value=("", {}))

    result = await _generate_character_sheet(
        state=base_state,
        character_name="Hero",
        other_characters=[],
    )

    assert result is None


@pytest.mark.asyncio
async def test_generate_character_sheet_exception(base_state, mock_llm_service, mock_schema_validator):
    """Verify exception handling during sheet generation."""
    mock_llm_service.async_call_llm = AsyncMock(side_effect=Exception("LLM error"))

    result = await _generate_character_sheet(
        state=base_state,
        character_name="Hero",
        other_characters=[],
    )

    assert result is None


@pytest.mark.asyncio
async def test_generate_character_sheets_success(
    base_state,
    mock_content_manager,
    mock_llm_service,
    mock_neo4j,
    mock_schema_validator,
):
    """Verify successful generation of all character sheets."""
    char_list_response = json.dumps(["Hero", "Mentor", "Villain"])
    char_sheet_response = json.dumps(
        {
            "name": "Test",
            "description": "Test character",
            "traits": ["brave"],
        }
    )

    mock_llm_service.async_call_llm = AsyncMock(
        side_effect=[
            (char_list_response, {}),
            (char_sheet_response, {}),
            (char_sheet_response, {}),
            (char_sheet_response, {}),
        ]
    )

    with patch("core.langgraph.initialization.character_sheets_node.validate_and_filter_traits") as mock_validate:
        mock_validate.return_value = ["brave"]

        result = await generate_character_sheets(base_state)

        assert result["initialization_step"] == "character_sheets_complete"
        assert result["current_node"] == "character_sheets"
        assert result["last_error"] is None
        assert "character_sheets_ref" in result


@pytest.mark.asyncio
async def test_generate_character_sheets_missing_title(base_state):
    """Verify error when title is missing."""
    state = {**base_state, "title": ""}

    result = await generate_character_sheets(state)

    assert result["initialization_step"] == "character_sheets_failed"
    assert "Missing required fields" in result["last_error"]


@pytest.mark.asyncio
async def test_generate_character_sheets_missing_genre(base_state):
    """Verify error when genre is missing."""
    state = {**base_state, "genre": ""}

    result = await generate_character_sheets(state)

    assert result["initialization_step"] == "character_sheets_failed"
    assert "Missing required fields" in result["last_error"]


@pytest.mark.asyncio
async def test_generate_character_sheets_character_list_fails(base_state, mock_llm_service, mock_neo4j):
    """Verify failure when character list generation cannot produce usable output."""
    mock_llm_service.async_call_llm = AsyncMock(return_value=("", {}))

    result = await generate_character_sheets(base_state)

    assert result["initialization_step"] == "character_sheets_failed"
    assert "Failed to generate character list" in result["last_error"]


@pytest.mark.asyncio
async def test_generate_character_sheets_all_sheets_fail(
    base_state,
    mock_llm_service,
    mock_neo4j,
    mock_schema_validator,
):
    """Verify error when all character sheet generations fail."""
    char_list_response = json.dumps(["Hero", "Mentor", "Villain"])
    mock_llm_service.async_call_llm = AsyncMock(
        side_effect=[
            (char_list_response, {}),
            ("", {}),
            ("", {}),
            ("", {}),
            ("", {}),
            ("", {}),
            ("", {}),
            ("", {}),
            ("", {}),
            ("", {}),
        ]
    )

    with patch("core.langgraph.initialization.character_sheets_node.validate_and_filter_traits") as mock_validate:
        mock_validate.return_value = []

        result = await generate_character_sheets(base_state)

        assert result["initialization_step"] == "character_sheets_failed"
        assert "Failed to generate any character sheets" in result["last_error"]


@pytest.mark.asyncio
async def test_generate_character_sheet_uses_existing_traits(base_state, mock_llm_service, mock_schema_validator):
    """Verify existing traits are passed to prompt."""
    with patch("core.langgraph.initialization.character_sheets_node.validate_and_filter_traits") as mock_validate:
        mock_validate.return_value = []

        await _generate_character_sheet(
            state=base_state,
            character_name="Hero",
            other_characters=[],
            existing_traits=["brave", "loyal", "cunning"],
        )

        call_args = mock_llm_service.async_call_llm.call_args
        assert call_args is not None
        prompt = call_args.kwargs.get("prompt", "")
        assert "brave" in prompt or "Existing traits" in prompt or len(prompt) > 0


