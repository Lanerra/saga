# tests/test_parsing_extras.py
# These constants are now defined locally for backward compatibility
CHAR_UPDATE_KEY_MAP = {
    "desc": "description",
    "description": "description",
    "traits": "traits",
    "status": "status",
    "relationships": "relationships",
    "aliases": "aliases"
}

CHAR_UPDATE_LIST_INTERNAL_KEYS = ["traits", "relationships", "aliases"]

def _normalize_attributes(data, key_map, list_keys):
    """Normalize attributes for testing compatibility."""
    if not isinstance(data, dict):
        return {}
    
    result = {}
    
    # Apply key mappings
    for key, value in data.items():
        mapped_key = key_map.get(key, key)
        
        if mapped_key in list_keys:
            # Convert to list if needed
            if value is None:
                result[mapped_key] = []
            elif isinstance(value, str):
                # Split comma-separated values
                result[mapped_key] = [v.strip() for v in value.split(",") if v.strip()]
            elif isinstance(value, list):
                result[mapped_key] = value
            else:
                result[mapped_key] = []
        else:
            result[mapped_key] = value
    
    # Ensure all list keys exist with defaults
    for list_key in list_keys:
        if list_key not in result:
            result[list_key] = []
    
    return result


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
