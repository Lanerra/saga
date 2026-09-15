# tests/test_chapter_outline_json_parsing.py
"""Test for chapter outline JSON parsing fix."""

import pytest

from core.langgraph.initialization.chapter_outline_node import _parse_chapter_outline


def test_parse_chapter_outline_with_object() -> None:
    """Test that _parse_chapter_outline correctly handles a JSON object response."""
    # Simulate a valid JSON object response from LLM
    response = """{
        "scene_description": "The protagonist enters the dark forest, sensing danger.",
        "key_beats": [
            "Protagonist hears rustling in the bushes",
            "A shadowy figure appears briefly",
            "Protagonist draws their weapon",
            "The figure vanishes without a trace"
        ],
        "plot_point": "The protagonist realizes they are being hunted."
    }"""

    result = _parse_chapter_outline(response, 1, 1)

    assert isinstance(result, dict)
    assert result["chapter_number"] == 1
    assert result["act_number"] == 1
    assert result["scene_description"] == "The protagonist enters the dark forest, sensing danger."
    assert len(result["key_beats"]) == 4
    assert result["plot_point"] == "The protagonist realizes they are being hunted."


def test_parse_chapter_outline_with_array_fallback() -> None:
    """An array must not be salvaged into a fabricated chapter outline."""
    response = """[
        "The protagonist enters the dark forest",
        "They hear rustling in the bushes",
        "A shadowy figure appears"
    ]"""

    with pytest.raises(ValueError):
        _parse_chapter_outline(response, 1, 1)


def test_parse_chapter_outline_with_invalid_json() -> None:
    """Free text cannot replace the required structured outline."""
    response = "This is not valid JSON but contains some text about a scene."

    with pytest.raises(ValueError):
        _parse_chapter_outline(response, 1, 1)


def test_parse_chapter_outline_missing_keys() -> None:
    """Missing required fields fail instead of receiving invented defaults."""
    response = """{
        "scene_description": "A scene description"
    }"""

    with pytest.raises(ValueError):
        _parse_chapter_outline(response, 1, 1)
