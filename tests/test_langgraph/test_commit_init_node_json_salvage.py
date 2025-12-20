import json

import pytest

from core.langgraph.initialization.commit_init_node import (
    _parse_character_extraction_response,
    _parse_world_items_extraction,
)


def test_character_extraction_parses_json_only_response() -> None:
    response = json.dumps(
        {
            "traits": ["brave", "loyal", "curious"],
            "status": "Active",
            "motivations": "Protect the village",
            "background": "Raised by a former ranger",
        }
    )
    parsed = _parse_character_extraction_response(response)
    assert parsed == {
        "traits": ["brave", "loyal", "curious"],
        "status": "Active",
        "motivations": "Protect the village",
        "background": "Raised by a former ranger",
    }


def test_character_extraction_salvages_embedded_json_object() -> None:
    embedded = json.dumps(
        {
            "traits": ["brave", "loyal", "curious"],
            "status": "Active",
            "motivations": "Protect the village",
            "background": "Raised by a former ranger",
        }
    )
    response = "Some commentary before.\n\n" + embedded + "\n\nSome commentary after."
    parsed = _parse_character_extraction_response(response)
    assert parsed["traits"] == ["brave", "loyal", "curious"]
    assert parsed["status"] == "Active"


def test_character_extraction_invalid_json_still_raises() -> None:
    response = "prefix\n" + '{"traits": ["brave"]' + "\npostfix"
    with pytest.raises(json.JSONDecodeError):
        _parse_character_extraction_response(response)


def test_world_items_extraction_parses_json_only_response() -> None:
    response = json.dumps(
        [
            {
                "name": "Castle Blackstone",
                "category": "location",
                "description": "An ancient fortress overlooking the pass",
            }
        ]
    )
    items = _parse_world_items_extraction(response)
    assert len(items) == 1
    assert items[0].name == "Castle Blackstone"
    assert items[0].category == "location"


def test_world_items_extraction_salvages_embedded_json_array() -> None:
    embedded = json.dumps(
        [
            {
                "name": "Castle Blackstone",
                "category": "location",
                "description": "An ancient fortress overlooking the pass",
            }
        ]
    )
    response = "Some commentary before.\n\n" + embedded + "\n\nSome commentary after."
    items = _parse_world_items_extraction(response)
    assert len(items) == 1
    assert items[0].name == "Castle Blackstone"


def test_world_items_extraction_invalid_json_still_raises() -> None:
    response = "prefix\n" + '[{"name": "Castle"' + "\npostfix"
    with pytest.raises(json.JSONDecodeError):
        _parse_world_items_extraction(response)