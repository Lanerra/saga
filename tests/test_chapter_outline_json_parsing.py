# tests/test_chapter_outline_json_parsing.py
"""Test for chapter outline JSON parsing fix."""

from core.langgraph.initialization.chapter_outline_node import _parse_chapter_outline


def test_parse_chapter_outline_with_object():
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


def test_parse_chapter_outline_with_array_fallback():
    """Test that _parse_chapter_outline handles array responses gracefully via fallback parsing."""
    # Simulate an array response (which shouldn't happen with the fixed prompt)
    # but test that the fallback parsing doesn't crash
    response = """[
        "The protagonist enters the dark forest",
        "They hear rustling in the bushes",
        "A shadowy figure appears"
    ]"""

    result = _parse_chapter_outline(response, 1, 1)

    assert isinstance(result, dict)
    assert result["chapter_number"] == 1
    assert result["act_number"] == 1
    # Fallback should use the full response as scene_description
    assert result["raw_text"] == response


def test_parse_chapter_outline_with_invalid_json():
    """Test that _parse_chapter_outline handles invalid JSON gracefully."""
    response = "This is not valid JSON but contains some text about a scene."

    result = _parse_chapter_outline(response, 1, 1)

    assert isinstance(result, dict)
    assert result["chapter_number"] == 1
    assert result["act_number"] == 1
    # Fallback should use the full response as scene_description
    assert result["scene_description"] == response


def test_parse_chapter_outline_missing_keys():
    """Test that _parse_chapter_outline handles missing keys with defaults."""
    response = """{
        "scene_description": "A scene description"
    }"""

    result = _parse_chapter_outline(response, 1, 1)

    assert isinstance(result, dict)
    assert result["chapter_number"] == 1
    assert result["act_number"] == 1
    assert result["scene_description"] == "A scene description"
    assert result["key_beats"] == []
    assert "Chapter 1 events" in result["plot_point"]
