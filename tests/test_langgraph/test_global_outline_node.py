import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.langgraph.initialization.global_outline_node import (
    ActOutline,
    GlobalOutlineSchema,
    _build_character_context_from_sheets,
    _extract_json_from_response,
    _fallback_parse_outline,
    _parse_global_outline,
    _validate_chapter_allocations,
    generate_global_outline,
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
    with patch("core.langgraph.initialization.global_outline_node.ContentManager") as mock:
        instance = MagicMock()
        instance.save_json.return_value = {
            "path": "mock/path/global_outline.json",
            "size_bytes": 3072,
            "version": 1,
        }
        mock.return_value = instance
        yield mock


@pytest.fixture
def sample_outline_json():
    """Sample valid outline JSON."""
    return {
        "act_count": 3,
        "acts": [
            {
                "act_number": 1,
                "title": "Setup",
                "summary": "Introduction of hero",
                "key_events": ["Hero meets mentor"],
                "chapters_start": 1,
                "chapters_end": 7,
            },
            {
                "act_number": 2,
                "title": "Confrontation",
                "summary": "Hero faces challenges",
                "key_events": ["Major battle"],
                "chapters_start": 8,
                "chapters_end": 14,
            },
            {
                "act_number": 3,
                "title": "Resolution",
                "summary": "Hero triumphs",
                "key_events": ["Final victory"],
                "chapters_start": 15,
                "chapters_end": 20,
            },
        ],
        "inciting_incident": "The kingdom is threatened",
        "midpoint": "Hero discovers the truth",
        "climax": "Final confrontation with villain",
        "resolution": "Peace is restored",
        "character_arcs": [
            {
                "character_name": "Hero",
                "starting_state": "Naive",
                "ending_state": "Wise",
                "key_moments": ["First failure", "Learning from mentor"],
            }
        ],
        "thematic_progression": "From innocence to experience",
        "pacing_notes": "Fast-paced action throughout",
    }


@pytest.fixture
def mock_llm_service(sample_outline_json):
    """Create a mock LLM service."""
    with patch("core.langgraph.initialization.global_outline_node.llm_service") as mock:
        mock.async_call_llm = AsyncMock(
            return_value=(
                json.dumps(sample_outline_json),
                {"prompt_tokens": 100, "completion_tokens": 50},
            )
        )
        yield mock


@pytest.fixture
def mock_get_character_sheets():
    """Mock get_character_sheets function."""
    with patch("core.langgraph.initialization.global_outline_node.get_character_sheets") as mock:
        mock.return_value = {
            "Hero": {
                "description": "A brave warrior",
                "is_protagonist": True,
            },
            "Mentor": {
                "description": "Wise guide",
                "is_protagonist": False,
            },
        }
        yield mock


@pytest.mark.asyncio
async def test_generate_global_outline_success(base_state, mock_content_manager, mock_llm_service, mock_get_character_sheets):
    """Verify successful generation of global outline."""
    state = {**base_state}

    result = await generate_global_outline(state)

    assert result["initialization_step"] == "global_outline_complete"
    assert result["current_node"] == "global_outline"
    assert result["last_error"] is None
    assert "global_outline_ref" in result


@pytest.mark.asyncio
async def test_generate_global_outline_empty_response(base_state, mock_content_manager, mock_llm_service, mock_get_character_sheets):
    """Verify error handling when LLM returns empty response."""
    mock_llm_service.async_call_llm = AsyncMock(return_value=("", {}))

    state = {**base_state}

    result = await generate_global_outline(state)

    assert result["initialization_step"] == "global_outline_failed"
    assert "empty global outline" in result["last_error"]


@pytest.mark.asyncio
async def test_generate_global_outline_exception(base_state, mock_content_manager, mock_llm_service, mock_get_character_sheets):
    """Verify exception handling during generation."""
    mock_llm_service.async_call_llm = AsyncMock(side_effect=Exception("LLM error"))

    state = {**base_state}

    result = await generate_global_outline(state)

    assert result["initialization_step"] == "global_outline_failed"
    assert "Error generating global outline" in result["last_error"]


@pytest.mark.asyncio
async def test_generate_global_outline_without_characters(base_state, mock_content_manager, mock_llm_service, mock_get_character_sheets):
    """Verify generation works without character sheets."""
    mock_get_character_sheets.return_value = {}

    state = {**base_state}

    result = await generate_global_outline(state)

    assert result["initialization_step"] == "global_outline_complete"


def test_build_character_context_from_sheets():
    """Verify building character context from sheets."""
    character_sheets = {
        "Hero": {
            "description": "A brave warrior on a quest.",
            "is_protagonist": True,
        },
        "Villain": {
            "description": "An evil sorcerer.",
            "is_protagonist": False,
        },
    }

    result = _build_character_context_from_sheets(character_sheets)

    assert "Hero" in result
    assert "Protagonist" in result
    assert "Villain" in result
    assert "Main Character" in result
    assert "brave warrior" in result


def test_build_character_context_empty():
    """Verify handling of empty character sheets."""
    result = _build_character_context_from_sheets({})

    assert result == "No characters defined yet."


def test_extract_json_from_response_with_code_blocks():
    """Verify JSON extraction from markdown code blocks."""
    response = """```json
{
    "key": "value"
}
```"""

    result = _extract_json_from_response(response)

    assert "key" in result
    assert "value" in result
    assert "```" not in result


def test_extract_json_from_response_raw_json():
    """Verify extraction of raw JSON without code blocks."""
    response = '{"key": "value", "number": 42}'

    result = _extract_json_from_response(response)

    assert result == response


def test_extract_json_from_response_mixed_content():
    """Verify JSON extraction from mixed content."""
    response = """Here is the outline:

{
    "key": "value"
}

That's the outline."""

    result = _extract_json_from_response(response)

    assert "{" in result
    assert "key" in result


def test_validate_chapter_allocations_valid():
    """Verify validation of correct chapter allocations."""
    outline = GlobalOutlineSchema(
        act_count=3,
        acts=[
            ActOutline(
                act_number=1,
                title="Act 1",
                summary="Setup",
                chapters_start=1,
                chapters_end=7,
            ),
            ActOutline(
                act_number=2,
                title="Act 2",
                summary="Confrontation",
                chapters_start=8,
                chapters_end=14,
            ),
            ActOutline(
                act_number=3,
                title="Act 3",
                summary="Resolution",
                chapters_start=15,
                chapters_end=20,
            ),
        ],
        inciting_incident="Test",
        midpoint="Test",
        climax="Test",
        resolution="Test",
        thematic_progression="Test",
    )

    errors = _validate_chapter_allocations(outline, 20)

    assert errors == []


def test_validate_chapter_allocations_no_acts():
    """Verify validation catches missing acts."""
    outline = GlobalOutlineSchema(
        act_count=3,
        acts=[],
        inciting_incident="Test",
        midpoint="Test",
        climax="Test",
        resolution="Test",
        thematic_progression="Test",
    )

    errors = _validate_chapter_allocations(outline, 20)

    assert len(errors) > 0
    assert "No acts defined" in errors[0]


def test_validate_chapter_allocations_overlapping_chapters():
    """Verify validation catches overlapping chapter ranges."""
    outline = GlobalOutlineSchema(
        act_count=2,
        acts=[
            ActOutline(
                act_number=1,
                title="Act 1",
                summary="Setup",
                chapters_start=1,
                chapters_end=10,
            ),
            ActOutline(
                act_number=2,
                title="Act 2",
                summary="Resolution",
                chapters_start=10,
                chapters_end=20,
            ),
        ],
        inciting_incident="Test",
        midpoint="Test",
        climax="Test",
        resolution="Test",
        thematic_progression="Test",
    )

    errors = _validate_chapter_allocations(outline, 20)

    assert any("multiple acts" in error for error in errors)


def test_validate_chapter_allocations_missing_chapters():
    """Verify validation catches missing chapter allocations."""
    outline = GlobalOutlineSchema(
        act_count=2,
        acts=[
            ActOutline(
                act_number=1,
                title="Act 1",
                summary="Setup",
                chapters_start=1,
                chapters_end=10,
            ),
            ActOutline(
                act_number=2,
                title="Act 2",
                summary="Resolution",
                chapters_start=15,
                chapters_end=20,
            ),
        ],
        inciting_incident="Test",
        midpoint="Test",
        climax="Test",
        resolution="Test",
        thematic_progression="Test",
    )

    errors = _validate_chapter_allocations(outline, 20)

    assert any("Missing chapter allocations" in error for error in errors)


def test_validate_chapter_allocations_extra_chapters():
    """Verify validation catches extra chapter allocations."""
    outline = GlobalOutlineSchema(
        act_count=1,
        acts=[
            ActOutline(
                act_number=1,
                title="Act 1",
                summary="Complete story",
                chapters_start=1,
                chapters_end=25,
            ),
        ],
        inciting_incident="Test",
        midpoint="Test",
        climax="Test",
        resolution="Test",
        thematic_progression="Test",
    )

    errors = _validate_chapter_allocations(outline, 20)

    assert any("Extra chapter allocations" in error for error in errors)


def test_validate_chapter_allocations_wrong_act_numbers():
    """Verify validation catches incorrect act numbering."""
    outline = GlobalOutlineSchema(
        act_count=3,
        acts=[
            ActOutline(
                act_number=1,
                title="Act 1",
                summary="Setup",
                chapters_start=1,
                chapters_end=7,
            ),
            ActOutline(
                act_number=3,
                title="Act 3",
                summary="Skip act 2",
                chapters_start=8,
                chapters_end=20,
            ),
        ],
        inciting_incident="Test",
        midpoint="Test",
        climax="Test",
        resolution="Test",
        thematic_progression="Test",
    )

    errors = _validate_chapter_allocations(outline, 20)

    assert any("Act numbers should be" in error for error in errors)


def test_parse_global_outline_valid_json(base_state, sample_outline_json):
    """Verify parsing of valid JSON outline."""
    response = json.dumps(sample_outline_json)

    result = _parse_global_outline(response, base_state)

    assert result["act_count"] == 3
    assert len(result["acts"]) == 3
    assert result["inciting_incident"] == "The kingdom is threatened"
    assert result["structure_type"] == "3-act"
    assert "raw_text" in result


def test_parse_global_outline_with_markdown(base_state, sample_outline_json):
    """Verify parsing of JSON wrapped in markdown."""
    response = f"""```json
{json.dumps(sample_outline_json)}
```"""

    result = _parse_global_outline(response, base_state)

    assert result["act_count"] == 3
    assert len(result["acts"]) == 3


def test_parse_global_outline_invalid_json_uses_fallback(base_state):
    """Verify fallback parsing when JSON is invalid."""
    response = "This is not valid JSON but contains Act 1, Act 2, and Act 3"

    result = _parse_global_outline(response, base_state)

    assert result["act_count"] == 3
    assert "Fallback parsing used" in result["validation_errors"][0]


def test_fallback_parse_outline_three_act(base_state):
    """Verify fallback parser detects 3-act structure."""
    response = "Act 1: Setup\nAct 2: Confrontation\nAct 3: Resolution"

    result = _fallback_parse_outline(response, base_state)

    assert result["act_count"] == 3
    assert result["structure_type"] == "3-act"
    assert result["raw_text"] == response


def test_fallback_parse_outline_five_act(base_state):
    """Verify fallback parser detects 5-act structure."""
    response = "Act 1, Act 2, Act 3, Act 4, Act 5 structure"

    result = _fallback_parse_outline(response, base_state)

    assert result["act_count"] == 5
    assert result["structure_type"] == "5-act"


def test_fallback_parse_outline_default_three_act(base_state):
    """Verify fallback parser defaults to 3-act."""
    response = "Just some outline text without act mentions"

    result = _fallback_parse_outline(response, base_state)

    assert result["act_count"] == 3


def test_fallback_parse_outline_roman_numerals(base_state):
    """Verify fallback parser recognizes Roman numeral acts."""
    response = "Act I: Setup, Act II: Rising Action, Act III: Climax, Act IV: Falling Action, Act V: Resolution"

    result = _fallback_parse_outline(response, base_state)

    assert result["act_count"] == 5


@pytest.mark.asyncio
async def test_generate_global_outline_uses_grammar(base_state, mock_content_manager, mock_llm_service, mock_get_character_sheets):
    """Verify generation uses GBNF grammar."""
    state = {**base_state}

    await generate_global_outline(state)

    call_args = mock_llm_service.async_call_llm.call_args
    assert call_args is not None
    kwargs = call_args.kwargs
    assert "grammar" in kwargs
    assert "root ::= global-outline" in kwargs["grammar"]


def test_parse_global_outline_preserves_all_fields(base_state, sample_outline_json):
    """Verify all fields from outline are preserved."""
    response = json.dumps(sample_outline_json)

    result = _parse_global_outline(response, base_state)

    assert result["inciting_incident"] == sample_outline_json["inciting_incident"]
    assert result["midpoint"] == sample_outline_json["midpoint"]
    assert result["climax"] == sample_outline_json["climax"]
    assert result["resolution"] == sample_outline_json["resolution"]
    assert result["thematic_progression"] == sample_outline_json["thematic_progression"]
    assert len(result["character_arcs"]) == len(sample_outline_json["character_arcs"])


def test_fallback_parse_outline_includes_validation_errors(base_state):
    """Verify fallback parser includes validation errors."""
    response = "Some outline text"

    result = _fallback_parse_outline(response, base_state)

    assert "validation_errors" in result
    assert len(result["validation_errors"]) > 0
    assert "Fallback parsing used" in result["validation_errors"][0]
