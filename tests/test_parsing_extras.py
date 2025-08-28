# tests/test_parsing_extras.py
from kg_maintainer.parsing import (
    CHAR_UPDATE_KEY_MAP,
    CHAR_UPDATE_LIST_INTERNAL_KEYS,
    _normalize_attributes,
)


def test_normalize_attributes_basic_mapping():
    data = {"desc": "Hero", "traits": "brave, kind"}
    result = _normalize_attributes(
        data, CHAR_UPDATE_KEY_MAP, CHAR_UPDATE_LIST_INTERNAL_KEYS
    )
    assert result["description"] == "Hero"
    assert result["traits"] == ["brave", "kind"]


def test_normalize_attributes_not_dict():
    assert (
        _normalize_attributes(
            "oops", CHAR_UPDATE_KEY_MAP, CHAR_UPDATE_LIST_INTERNAL_KEYS
        )
        == {}
    )


def test_normalize_attributes_defaults_and_none():
    data = {"traits": None}
    result = _normalize_attributes(
        data, CHAR_UPDATE_KEY_MAP, CHAR_UPDATE_LIST_INTERNAL_KEYS
    )
    assert result["traits"] == []
    assert result["relationships"] == []
    assert result["aliases"] == []
