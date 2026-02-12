# tests/test_langgraph/test_act_outlines_node.py
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.langgraph.initialization.act_outlines_node import (
    _generate_single_act_outline,
    _get_act_role,
    generate_act_outlines,
)
from core.langgraph.initialization.chapter_outline_node import build_character_summary
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
    with patch("core.langgraph.initialization.act_outlines_node.ContentManager") as mock:
        instance = MagicMock()
        instance.save_json.return_value = {
            "path": "mock/path/act_outlines.json",
            "size_bytes": 2048,
            "version": 1,
        }
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    with patch("core.langgraph.initialization.act_outlines_node.llm_service") as mock:

        async def side_effect(*args, **kwargs):
            prompt = kwargs.get("prompt", "")
            act_match = re.search(r"Act:\s*(\d+)\s*of\s*(\d+)", prompt)
            chapters_match = re.search(r"Chapters in act:\s*~(\d+)", prompt)

            if not act_match:
                raise ValueError("Test fixture expected act_number in prompt")
            if not chapters_match:
                raise ValueError("Test fixture expected chapters_in_act in prompt")

            act_number = int(act_match.group(1))
            total_acts = int(act_match.group(2))
            chapters_in_act = int(chapters_match.group(1))
            act_role = _get_act_role(act_number, total_acts)

            data = {
                "act_number": act_number,
                "total_acts": total_acts,
                "act_role": act_role,
                "chapters_in_act": chapters_in_act,
                "sections": {
                    "act_summary": "A concise summary of the act.",
                    "opening_situation": "The opening situation establishes the problem.",
                    "key_events": [
                        {"sequence": 1, "event": "Event 1", "cause": "Cause 1", "effect": "Effect 1"},
                        {"sequence": 2, "event": "Event 2", "cause": "Cause 2", "effect": "Effect 2"},
                        {"sequence": 3, "event": "Event 3", "cause": "Cause 3", "effect": "Effect 3"},
                        {"sequence": 4, "event": "Event 4", "cause": "Cause 4", "effect": "Effect 4"},
                        {"sequence": 5, "event": "Event 5", "cause": "Cause 5", "effect": "Effect 5"},
                    ],
                    "character_development": "Internal conflict sharpens through choices.",
                    "stakes_and_tension": "Pressure escalates as consequences compound.",
                    "act_ending_turn": "A decisive shift forces a new direction.",
                    "thematic_thread": "Theme appears through recurring imagery and action.",
                    "pacing_notes": "Early chapters slower; later chapters faster.",
                },
            }

            return (data, {"prompt_tokens": 100, "completion_tokens": 50})

        mock.async_call_llm_json_object = AsyncMock(side_effect=side_effect)
        yield mock


@pytest.fixture
def mock_get_functions():
    """Mock the content getter functions."""
    with patch("core.langgraph.initialization.act_outlines_node.get_global_outline") as mock_global, patch("core.langgraph.initialization.act_outlines_node.get_character_sheets") as mock_chars:
        mock_global.return_value = {
            "act_count": 3,
            "raw_text": "A three-act story about a hero's journey.",
        }

        mock_chars.return_value = {
            "Hero": {
                "description": "A brave warrior",
                "is_protagonist": True,
            },
            "Mentor": {
                "description": "Wise guide",
                "is_protagonist": False,
            },
        }

        yield {
            "global": mock_global,
            "chars": mock_chars,
        }


@pytest.mark.asyncio
async def test_generate_act_outlines_success(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    """Verify successful generation of act outlines."""
    state = {**base_state, "total_chapters": 21}

    result = await generate_act_outlines(state)

    assert result["initialization_step"] == "act_outlines_complete"
    assert result["current_node"] == "act_outlines"
    assert result["last_error"] is None
    assert "act_outlines_ref" in result
    assert result["act_outlines_ref"]["size_bytes"] == 2048
    assert mock_llm_service.async_call_llm_json_object.call_count == 3


@pytest.mark.asyncio
async def test_generate_act_outlines_missing_global_outline(base_state, mock_content_manager, mock_get_functions):
    """Verify error when global outline is missing."""
    mock_get_functions["global"].return_value = None

    state = {**base_state}

    result = await generate_act_outlines(state)

    assert result["initialization_step"] == "act_outlines_failed"
    assert result["current_node"] == "act_outlines"
    assert "No global outline" in result["last_error"]


@pytest.mark.asyncio
async def test_generate_act_outlines_five_acts(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    """Verify generation for five-act structure."""
    mock_get_functions["global"].return_value = {
        "act_count": 5,
        "raw_text": "A five-act story.",
    }

    state = {**base_state, "total_chapters": 20}

    result = await generate_act_outlines(state)

    assert result["initialization_step"] == "act_outlines_complete"
    assert mock_llm_service.async_call_llm_json_object.call_count == 5


@pytest.mark.asyncio
async def test_generate_act_outlines_single_act_fails(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    """Verify handling when single act generation fails."""
    call_count = [0]
    original_side_effect = mock_llm_service.async_call_llm_json_object.side_effect

    async def mock_call(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 2:
            raise ValueError("LLM returned invalid JSON")
        return await original_side_effect(*args, **kwargs)

    mock_llm_service.async_call_llm_json_object = AsyncMock(side_effect=mock_call)

    state = {**base_state}

    result = await generate_act_outlines(state)

    assert result["initialization_step"] == "act_outlines_complete"
    assert result["current_node"] == "act_outlines"


@pytest.mark.asyncio
async def test_generate_act_outlines_all_fail(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    """Verify error when all act generations fail."""
    mock_llm_service.async_call_llm_json_object = AsyncMock(side_effect=ValueError("LLM returned invalid JSON"))

    state = {**base_state}

    result = await generate_act_outlines(state)

    assert result["initialization_step"] == "act_outlines_failed"
    assert "Failed to generate any act outlines" in result["last_error"]


@pytest.mark.asyncio
async def test_generate_single_act_outline_success(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    """Verify successful generation of single act outline."""
    state = {**base_state}

    result = await _generate_single_act_outline(
        state=state,
        act_number=1,
        total_acts=3,
        chapters_in_act=7,
    )

    assert result is not None
    assert result["act_number"] == 1
    assert result["raw_text"] is not None
    assert "1. Act Summary" in result["raw_text"]
    assert "3. Key Events" in result["raw_text"]
    assert result["chapters_in_act"] == 7
    assert result["act_role"] == "Setup/Introduction"
    assert result["generated_at"] == "initialization"
    assert "sections" in result
    assert isinstance(result["sections"], dict)


@pytest.mark.asyncio
async def test_generate_single_act_outline_empty_response(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    """Verify JSON wrapper contract violation is surfaced for direct callers."""
    mock_llm_service.async_call_llm_json_object = AsyncMock(side_effect=ValueError("LLM returned invalid JSON"))

    state = {**base_state}

    with pytest.raises(ValueError, match="invalid JSON"):
        await _generate_single_act_outline(
            state=state,
            act_number=1,
            total_acts=3,
            chapters_in_act=7,
        )


@pytest.mark.asyncio
async def test_generate_single_act_outline_exception(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    """Verify exception handling during generation."""
    mock_llm_service.async_call_llm_json_object = AsyncMock(side_effect=Exception("LLM error"))

    state = {**base_state}

    result = await _generate_single_act_outline(
        state=state,
        act_number=1,
        total_acts=3,
        chapters_in_act=7,
    )

    assert result is None


@pytest.mark.asyncio
async def test_generate_single_act_outline_rejects_extra_keys(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    state = {**base_state}

    data = {
        "act_number": 1,
        "total_acts": 3,
        "act_role": "Setup/Introduction",
        "chapters_in_act": 7,
        "sections": {
            "act_summary": "Summary.",
            "opening_situation": "Opening.",
            "key_events": [
                {"sequence": 1, "event": "Event 1", "cause": "Cause 1", "effect": "Effect 1"},
                {"sequence": 2, "event": "Event 2", "cause": "Cause 2", "effect": "Effect 2"},
                {"sequence": 3, "event": "Event 3", "cause": "Cause 3", "effect": "Effect 3"},
                {"sequence": 4, "event": "Event 4", "cause": "Cause 4", "effect": "Effect 4"},
                {"sequence": 5, "event": "Event 5", "cause": "Cause 5", "effect": "Effect 5"},
            ],
            "character_development": "Development.",
            "stakes_and_tension": "Stakes.",
            "act_ending_turn": "Turn.",
            "thematic_thread": "Theme.",
            "pacing_notes": "Pacing.",
        },
        "unexpected": "nope",
    }
    mock_llm_service.async_call_llm_json_object = AsyncMock(return_value=(data, {}))

    with pytest.raises(ValueError, match="Extra inputs are not permitted|extra_forbidden|unexpected"):
        await _generate_single_act_outline(
            state=state,
            act_number=1,
            total_acts=3,
            chapters_in_act=7,
        )


@pytest.mark.asyncio
async def test_generate_single_act_outline_rejects_key_events_wrong_length(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    state = {**base_state}

    data = {
        "act_number": 1,
        "total_acts": 3,
        "act_role": "Setup/Introduction",
        "chapters_in_act": 7,
        "sections": {
            "act_summary": "Summary.",
            "opening_situation": "Opening.",
            "key_events": [
                {"sequence": 1, "event": "Event 1", "cause": "Cause 1", "effect": "Effect 1"},
                {"sequence": 2, "event": "Event 2", "cause": "Cause 2", "effect": "Effect 2"},
                {"sequence": 3, "event": "Event 3", "cause": "Cause 3", "effect": "Effect 3"},
                {"sequence": 4, "event": "Event 4", "cause": "Cause 4", "effect": "Effect 4"},
            ],
            "character_development": "Development.",
            "stakes_and_tension": "Stakes.",
            "act_ending_turn": "Turn.",
            "thematic_thread": "Theme.",
            "pacing_notes": "Pacing.",
        },
    }
    mock_llm_service.async_call_llm_json_object = AsyncMock(return_value=(data, {}))

    with pytest.raises(ValueError, match="key_events must contain 5-7"):
        await _generate_single_act_outline(
            state=state,
            act_number=1,
            total_acts=3,
            chapters_in_act=7,
        )


@pytest.mark.asyncio
async def test_generate_single_act_outline_rejects_cross_check_mismatch(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    state = {**base_state}

    data = {
        "act_number": 2,
        "total_acts": 3,
        "act_role": "Setup/Introduction",
        "chapters_in_act": 7,
        "sections": {
            "act_summary": "Summary.",
            "opening_situation": "Opening.",
            "key_events": [
                {"sequence": 1, "event": "Event 1", "cause": "Cause 1", "effect": "Effect 1"},
                {"sequence": 2, "event": "Event 2", "cause": "Cause 2", "effect": "Effect 2"},
                {"sequence": 3, "event": "Event 3", "cause": "Cause 3", "effect": "Effect 3"},
                {"sequence": 4, "event": "Event 4", "cause": "Cause 4", "effect": "Effect 4"},
                {"sequence": 5, "event": "Event 5", "cause": "Cause 5", "effect": "Effect 5"},
            ],
            "character_development": "Development.",
            "stakes_and_tension": "Stakes.",
            "act_ending_turn": "Turn.",
            "thematic_thread": "Theme.",
            "pacing_notes": "Pacing.",
        },
    }
    mock_llm_service.async_call_llm_json_object = AsyncMock(return_value=(data, {}))

    with pytest.raises(ValueError, match="act_number mismatch"):
        await _generate_single_act_outline(
            state=state,
            act_number=1,
            total_acts=3,
            chapters_in_act=7,
        )


@pytest.mark.asyncio
async def test_generate_single_act_outline_rejects_missing_required_keys(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    state = {**base_state}

    data = {
        "act_number": 1,
        "total_acts": 3,
        "act_role": "Setup/Introduction",
        "chapters_in_act": 7,
        "sections": {
            "act_summary": "Summary.",
            "opening_situation": "Opening.",
            "key_events": [
                {"sequence": 1, "event": "Event 1", "cause": "Cause 1", "effect": "Effect 1"},
                {"sequence": 2, "event": "Event 2", "cause": "Cause 2", "effect": "Effect 2"},
                {"sequence": 3, "event": "Event 3", "cause": "Cause 3", "effect": "Effect 3"},
                {"sequence": 4, "event": "Event 4", "cause": "Cause 4", "effect": "Effect 4"},
                {"sequence": 5, "event": "Event 5", "cause": "Cause 5", "effect": "Effect 5"},
            ],
            "character_development": "Development.",
            "stakes_and_tension": "Stakes.",
            "act_ending_turn": "Turn.",
            "thematic_thread": "Theme.",
        },
    }
    mock_llm_service.async_call_llm_json_object = AsyncMock(return_value=(data, {}))

    with pytest.raises(ValueError, match="Field required|pacing_notes"):
        await _generate_single_act_outline(
            state=state,
            act_number=1,
            total_acts=3,
            chapters_in_act=7,
        )


@pytest.mark.asyncio
async def test_generate_single_act_outline_rejects_wrong_types(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    state = {**base_state}

    data = {
        "act_number": "1",
        "total_acts": 3,
        "act_role": "Setup/Introduction",
        "chapters_in_act": 7,
        "sections": {
            "act_summary": "Summary.",
            "opening_situation": "Opening.",
            "key_events": [
                {"sequence": 1, "event": "Event 1", "cause": "Cause 1", "effect": "Effect 1"},
                {"sequence": 2, "event": "Event 2", "cause": "Cause 2", "effect": "Effect 2"},
                {"sequence": 3, "event": "Event 3", "cause": "Cause 3", "effect": "Effect 3"},
                {"sequence": 4, "event": "Event 4", "cause": "Cause 4", "effect": "Effect 4"},
                {"sequence": 5, "event": "Event 5", "cause": "Cause 5", "effect": "Effect 5"},
            ],
            "character_development": "Development.",
            "stakes_and_tension": "Stakes.",
            "act_ending_turn": "Turn.",
            "thematic_thread": "Theme.",
            "pacing_notes": "Pacing.",
        },
    }
    mock_llm_service.async_call_llm_json_object = AsyncMock(return_value=(data, {}))

    with pytest.raises(ValueError, match="act_number|int|integer"):
        await _generate_single_act_outline(
            state=state,
            act_number=1,
            total_acts=3,
            chapters_in_act=7,
        )


def test_get_act_role_three_act_structure():
    """Verify act role determination for 3-act structure."""
    assert _get_act_role(1, 3) == "Setup/Introduction"
    assert _get_act_role(2, 3) == "Confrontation/Rising Action"
    assert _get_act_role(3, 3) == "Resolution/Climax"


def test_get_act_role_five_act_structure():
    """Verify act role determination for 5-act structure."""
    assert _get_act_role(1, 5) == "Setup/Introduction"
    assert _get_act_role(2, 5) == "Rising Action"
    assert _get_act_role(3, 5) == "Midpoint/Crisis"
    assert _get_act_role(4, 5) == "Falling Action"
    assert _get_act_role(5, 5) == "Resolution/Climax"


def test_get_act_role_four_act_structure():
    """Verify act role determination for non-standard act count."""
    assert _get_act_role(1, 4) == "Setup/Introduction"
    assert _get_act_role(2, 4) == "Development"
    assert _get_act_role(3, 4) == "Development"
    assert _get_act_role(4, 4) == "Resolution/Climax"


def test_build_character_summary_with_descriptions():
    """Verify character summary includes descriptions when requested."""
    character_sheets = {
        "Hero": {
            "description": "A brave warrior on a quest.",
            "is_protagonist": True,
        },
        "Villain": {
            "description": "An evil sorcerer seeking power.",
            "is_protagonist": False,
        },
    }

    result = build_character_summary(character_sheets, include_description=True)

    assert "Hero" in result
    assert "Protagonist" in result
    assert "Villain" in result
    assert "Character" in result
    assert "A brave warrior" in result


def test_build_character_summary_empty():
    """Verify character summary with no characters."""
    result = build_character_summary({})

    assert result == "No characters defined."


def test_build_character_summary_full_description():
    """Verify character summary includes full description text."""
    character_sheets = {
        "Hero": {
            "description": "A very detailed description of the hero with backstory.",
            "is_protagonist": True,
        },
    }

    result = build_character_summary(character_sheets, include_description=True)

    assert "Hero" in result
    assert "A very detailed description" in result


@pytest.mark.asyncio
async def test_generate_act_outlines_uses_character_context(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    """Verify act outline generation uses character context."""
    state = {**base_state}

    await generate_act_outlines(state)

    call_args = mock_llm_service.async_call_llm_json_object.call_args
    assert call_args is not None
    prompt = call_args.kwargs.get("prompt", "")
    assert "Hero" in prompt or "prompt" in call_args.kwargs


@pytest.mark.asyncio
async def test_generate_act_outlines_uses_global_outline(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    """Verify act outline generation uses global outline."""
    state = {**base_state}

    await generate_act_outlines(state)

    call_args = mock_llm_service.async_call_llm_json_object.call_args
    assert call_args is not None


@pytest.mark.asyncio
async def test_generate_single_act_outline_different_acts(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    """Verify generation works for different act numbers."""
    state = {**base_state}

    for act_num in range(1, 6):
        result = await _generate_single_act_outline(
            state=state,
            act_number=act_num,
            total_acts=5,
            chapters_in_act=4,
        )

        assert result is not None
        assert result["act_number"] == act_num
        assert result["chapters_in_act"] == 4


@pytest.mark.asyncio
async def test_generate_act_outlines_calculates_chapters_per_act(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    """Verify correct calculation of chapters per act."""
    state = {**base_state, "total_chapters": 18}
    mock_get_functions["global"].return_value = {
        "act_count": 3,
        "raw_text": "Global outline",
    }

    await generate_act_outlines(state)

    assert mock_llm_service.async_call_llm_json_object.call_count == 3


@pytest.mark.asyncio
async def test_generate_act_outlines_remainder_distribution(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    """
    total_chapters not divisible by act_count should:
    - cover all chapters exactly once
    - distribute remainder to early acts (sizes differ by at most 1)
    """
    # 20 chapters, 3 acts => 7,7,6 with ranges: 1-7, 8-14, 15-20
    state = {**base_state, "total_chapters": 20}
    mock_get_functions["global"].return_value = {
        "act_count": 3,
        "raw_text": "Global outline",
    }

    await generate_act_outlines(state)

    cm_instance = mock_content_manager.return_value
    saved_act_outlines = cm_instance.save_json.call_args.args[0]

    assert saved_act_outlines["format_version"] == 2
    assert isinstance(saved_act_outlines["acts"], list)

    acts_by_number = {a["act_number"]: a for a in saved_act_outlines["acts"]}

    assert acts_by_number[1]["chapters_in_act"] == 7
    assert acts_by_number[1]["chapters_start"] == 1
    assert acts_by_number[1]["chapters_end"] == 7

    assert acts_by_number[2]["chapters_in_act"] == 7
    assert acts_by_number[2]["chapters_start"] == 8
    assert acts_by_number[2]["chapters_end"] == 14

    assert acts_by_number[3]["chapters_in_act"] == 6
    assert acts_by_number[3]["chapters_start"] == 15
    assert acts_by_number[3]["chapters_end"] == 20


@pytest.mark.asyncio
async def test_generate_act_outlines_act_count_greater_than_chapters(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    """
    act_count > total_chapters should not crash; later acts should receive empty ranges.

    Example: 2 chapters, 5 acts => ranges:
      act1: 1-1 (1)
      act2: 2-2 (1)
      act3: 3-2 (0) empty
      act4: 3-2 (0) empty
      act5: 3-2 (0) empty
    """
    state = {**base_state, "total_chapters": 2}
    mock_get_functions["global"].return_value = {
        "act_count": 5,
        "raw_text": "Global outline",
    }

    await generate_act_outlines(state)

    assert mock_llm_service.async_call_llm_json_object.call_count == 5

    cm_instance = mock_content_manager.return_value
    saved_act_outlines = cm_instance.save_json.call_args.args[0]

    assert saved_act_outlines["format_version"] == 2
    assert isinstance(saved_act_outlines["acts"], list)

    acts_by_number = {a["act_number"]: a for a in saved_act_outlines["acts"]}

    assert acts_by_number[1]["chapters_in_act"] == 1
    assert acts_by_number[1]["chapters_start"] == 1
    assert acts_by_number[1]["chapters_end"] == 1

    assert acts_by_number[2]["chapters_in_act"] == 1
    assert acts_by_number[2]["chapters_start"] == 2
    assert acts_by_number[2]["chapters_end"] == 2

    # Empty ranges: end < start and chapters_in_act == 0
    for act_num in [3, 4, 5]:
        assert acts_by_number[act_num]["chapters_in_act"] == 0
        assert acts_by_number[act_num]["chapters_start"] == 3
        assert acts_by_number[act_num]["chapters_end"] == 2


@pytest.mark.asyncio
async def test_generate_act_outlines_uses_explicit_global_ranges(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    """If the global outline provides explicit chapters_start/chapters_end, use them."""
    state = {**base_state, "total_chapters": 6}
    mock_get_functions["global"].return_value = {
        "act_count": 3,
        "acts": [
            {"act_number": 1, "chapters_start": 1, "chapters_end": 2},
            {"act_number": 2, "chapters_start": 3, "chapters_end": 5},
            {"act_number": 3, "chapters_start": 6, "chapters_end": 6},
        ],
        "raw_text": "Global outline",
    }

    await generate_act_outlines(state)

    cm_instance = mock_content_manager.return_value
    saved_act_outlines = cm_instance.save_json.call_args.args[0]

    assert saved_act_outlines["format_version"] == 2
    assert isinstance(saved_act_outlines["acts"], list)

    acts_by_number = {a["act_number"]: a for a in saved_act_outlines["acts"]}

    assert acts_by_number[1]["chapters_in_act"] == 2
    assert acts_by_number[1]["chapters_start"] == 1
    assert acts_by_number[1]["chapters_end"] == 2

    assert acts_by_number[2]["chapters_in_act"] == 3
    assert acts_by_number[2]["chapters_start"] == 3
    assert acts_by_number[2]["chapters_end"] == 5

    assert acts_by_number[3]["chapters_in_act"] == 1
    assert acts_by_number[3]["chapters_start"] == 6
    assert acts_by_number[3]["chapters_end"] == 6


@pytest.mark.asyncio
async def test_generate_act_outlines_without_character_sheets(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    """Verify generation works without character sheets."""
    mock_get_functions["chars"].return_value = {}

    state = {**base_state}

    result = await generate_act_outlines(state)

    assert result["initialization_step"] == "act_outlines_complete"
    assert result["current_node"] == "act_outlines"


@pytest.mark.asyncio
async def test_generate_act_outlines_stores_all_acts(base_state, mock_content_manager, mock_llm_service, mock_get_functions):
    """Verify all successfully generated acts are stored."""
    responses = []
    for act_number in range(1, 4):
        responses.append(
            (
                {
                    "act_number": act_number,
                    "total_acts": 3,
                    "act_role": _get_act_role(act_number, 3),
                    "chapters_in_act": 7,
                    "sections": {
                        "act_summary": "Summary.",
                        "opening_situation": "Opening.",
                        "key_events": [
                            {"sequence": 1, "event": "Event 1", "cause": "Cause 1", "effect": "Effect 1"},
                            {"sequence": 2, "event": "Event 2", "cause": "Cause 2", "effect": "Effect 2"},
                            {"sequence": 3, "event": "Event 3", "cause": "Cause 3", "effect": "Effect 3"},
                            {"sequence": 4, "event": "Event 4", "cause": "Cause 4", "effect": "Effect 4"},
                            {"sequence": 5, "event": "Event 5", "cause": "Cause 5", "effect": "Effect 5"},
                        ],
                        "character_development": "Development.",
                        "stakes_and_tension": "Stakes.",
                        "act_ending_turn": "Turn.",
                        "thematic_thread": "Theme.",
                        "pacing_notes": "Pacing.",
                    },
                },
                {"prompt_tokens": 100, "completion_tokens": 50},
            )
        )

    mock_llm_service.async_call_llm_json_object = AsyncMock(side_effect=responses)

    state = {**base_state}

    result = await generate_act_outlines(state)

    assert result["initialization_step"] == "act_outlines_complete"
    assert mock_content_manager.return_value.save_json.called
