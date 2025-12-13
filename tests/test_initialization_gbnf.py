import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from core.langgraph.initialization.character_sheets_node import (
    _parse_character_sheet_response,
    generate_character_sheets,
)
from core.langgraph.initialization.global_outline_node import generate_global_outline

# Mock State
MOCK_STATE = {
    "title": "Test Story",
    "genre": "Sci-Fi",
    "theme": "Hope",
    "setting": "Mars",
    "protagonist_name": "Commander Shepard",
    "large_model": "test-model",
    "project_dir": "/tmp/test_project",
}

# Mock LLM Responses
GLOBAL_OUTLINE_JSON = """
{
    "act_count": 3,
    "acts": [
        {"act_number": 1, "title": "Arrival", "summary": "Landing on Mars", "key_events": ["Crash"], "chapters_start": 1, "chapters_end": 5, "chapters": 5},
        {"act_number": 2, "title": "Survival", "summary": "Surviving Mars", "key_events": ["Water found"], "chapters_start": 6, "chapters_end": 15, "chapters": 10},
        {"act_number": 3, "title": "Rescue", "summary": "Escape Mars", "key_events": ["Signal sent"], "chapters_start": 16, "chapters_end": 20, "chapters": 5}
    ],
    "inciting_incident": "Ship crash",
    "midpoint": "Discovery of alien ruins",
    "climax": "Fighting the storm",
    "resolution": "Returning home",
    "character_arcs": [
        {"character_name": "Commander Shepard", "starting_state": "Confident", "ending_state": "Humble", "key_moments": ["First failure"]}
    ],
    "thematic_progression": "From arrogance to humility",
    "pacing_notes": "Fast paced"
}
"""

CHARACTER_LIST_RESPONSE = "Commander Shepard\nGarrus\nLiara"

CHARACTER_SHEET_JSON = """
{
    "name": "Commander Shepard",
    "description": "A veteran soldier.",
    "traits": ["Brave", "Leader"],
    "status": "Active",
    "motivations": "Save the galaxy",
    "background": "Born on Earth.",
    "skills": ["Shooting", "Diplomacy"],
    "relationships": {
        "Garrus": "Loyal friend",
        "Liara": "Trusted advisor"
    },
    "internal_conflict": "Burden of command"
}
"""


@pytest.mark.asyncio
async def test_global_outline_gbnf_integration():
    """Verify global_outline_node correctly loads and modifies grammar."""

    with patch("core.langgraph.initialization.global_outline_node.llm_service") as mock_llm:
        mock_llm.async_call_llm = AsyncMock(return_value=(GLOBAL_OUTLINE_JSON, {"tokens": 100}))

        # Mock ContentManager to avoid file I/O
        with patch("core.langgraph.initialization.global_outline_node.ContentManager") as mock_cm:
            mock_cm_instance = mock_cm.return_value
            mock_cm_instance.save_json.return_value = {
                "path": "mock/path",
                "size_bytes": 100,
            }

            # Mock get_character_sheets to return empty dict
            with patch(
                "core.langgraph.initialization.global_outline_node.get_character_sheets",
                return_value={},
            ):
                result_state = await generate_global_outline(MOCK_STATE.copy())

                # Verify LLM call arguments
                call_args = mock_llm.async_call_llm.call_args
                assert call_args is not None
                kwargs = call_args.kwargs

                # Check grammar parameter
                assert "grammar" in kwargs
                grammar = kwargs["grammar"]
                assert "root ::= global-outline" in grammar
                assert "global-outline ::=" in grammar

                # Verify parsing
                assert result_state["initialization_step"] == "global_outline_complete"
                assert "global_outline_ref" in result_state


@pytest.mark.asyncio
async def test_character_sheets_gbnf_integration():
    """Verify character_sheets_node correctly loads and modifies grammar."""

    with patch("core.langgraph.initialization.character_sheets_node.llm_service") as mock_llm:
        # Mock responses: 1 for list, 1 for sheet (simplified for 1 char)
        mock_llm.async_call_llm = AsyncMock(
            side_effect=[
                (CHARACTER_LIST_RESPONSE, {"tokens": 50}),
                (CHARACTER_SHEET_JSON, {"tokens": 200}),
                (CHARACTER_SHEET_JSON, {"tokens": 200}),
                (CHARACTER_SHEET_JSON, {"tokens": 200}),
            ]
        )

        # Mock ContentManager
        with patch("core.langgraph.initialization.character_sheets_node.ContentManager") as mock_cm:
            mock_cm_instance = mock_cm.return_value
            mock_cm_instance.save_json.return_value = {
                "path": "mock/path",
                "size_bytes": 100,
            }

            result_state = await generate_character_sheets(MOCK_STATE.copy())

            # Verify LLM call for sheet generation
            # First call is list, second is sheet
            assert mock_llm.async_call_llm.call_count >= 2

            # Check arguments of the last call (sheet generation)
            call_args = mock_llm.async_call_llm.call_args
            kwargs = call_args.kwargs

            # Check grammar parameter
            assert "grammar" in kwargs
            grammar = kwargs["grammar"]
            assert "root ::= character-sheet" in grammar
            assert "character-sheet ::=" in grammar

            # Verify parsing logic (unit test for parser)
            parsed = _parse_character_sheet_response(CHARACTER_SHEET_JSON, "Commander Shepard")
            assert parsed["name"] == "Commander Shepard"
            assert parsed["status"] == "Active"
            assert len(parsed["traits"]) == 2
            assert parsed["relationships"]["Garrus"]["description"] == "Loyal friend"


if __name__ == "__main__":
    import sys

    # Run tests manually if executed as script
    try:
        asyncio.run(test_global_outline_gbnf_integration())
        print("test_global_outline_gbnf_integration passed")
        asyncio.run(test_character_sheets_gbnf_integration())
        print("test_character_sheets_gbnf_integration passed")
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
