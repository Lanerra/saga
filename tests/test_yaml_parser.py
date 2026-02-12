# tests/test_yaml_parser.py
import os

import pytest
import yaml

from utils.common import (
    load_yaml_file,
    normalize_keys_recursive,
)


@pytest.fixture
def yaml_test_dir(tmp_path):
    """Create temporary YAML test files."""
    valid_content = {
        "Novel Concept": {"Title": "Test Novel", "Genre": "Sci-Fi"},
        "protagonist_traits": ["Brave", "Smart"],
    }
    valid_path = tmp_path / "valid.yaml"
    with open(valid_path, "w", encoding="utf-8") as f:
        yaml.dump(valid_content, f)

    malformed_path = tmp_path / "malformed.yaml"
    with open(malformed_path, "w", encoding="utf-8") as f:
        f.write("Novel Concept: Title: Test Novel\nGenre: [Sci-Fi")

    empty_path = tmp_path / "empty.yaml"
    with open(empty_path, "w", encoding="utf-8") as f:
        f.write("")

    non_dict_root_path = tmp_path / "non_dict_root.yaml"
    with open(non_dict_root_path, "w", encoding="utf-8") as f:
        f.write("- item1\n- item2")

    return {
        "valid": str(valid_path),
        "malformed": str(malformed_path),
        "empty": str(empty_path),
        "non_dict_root": str(non_dict_root_path),
    }


class TestYamlParsing:
    def test_load_valid_yaml_normalized_keys(self, yaml_test_dir) -> None:
        data = load_yaml_file(yaml_test_dir["valid"], normalize_keys=True)
        assert "novel_concept" in data
        assert data["novel_concept"]["title"] == "Test Novel"
        assert data["protagonist_traits"] == ["Brave", "Smart"]

    def test_load_valid_yaml_raw_keys(self, yaml_test_dir) -> None:
        data = load_yaml_file(yaml_test_dir["valid"], normalize_keys=False)
        assert "Novel Concept" in data
        assert data["Novel Concept"]["Title"] == "Test Novel"

    def test_load_non_existent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_yaml_file("non_existent.yaml")

    def test_load_malformed_yaml_raises(self, yaml_test_dir) -> None:
        with pytest.raises(yaml.YAMLError):
            load_yaml_file(yaml_test_dir["malformed"])

    def test_load_empty_yaml(self, yaml_test_dir) -> None:
        data = load_yaml_file(yaml_test_dir["empty"])
        assert data == {}

    def test_load_non_dict_root_yaml_raises(self, yaml_test_dir) -> None:
        with pytest.raises(ValueError, match="must have a dictionary as its root element"):
            load_yaml_file(yaml_test_dir["non_dict_root"])

    def test_load_non_yaml_extension_raises(self) -> None:
        with pytest.raises(ValueError, match="not a YAML file"):
            load_yaml_file("some_file.txt")

    def test_normalize_keys_recursive(self) -> None:
        data = {
            "First Key": {"Second Level Key": "value1"},
            "Another Top Key": [{"List Key One": 1}, {"List Key Two": 2}],
        }
        normalized = normalize_keys_recursive(data)
        expected = {
            "first_key": {"second_level_key": "value1"},
            "another_top_key": [{"list_key_one": 1}, {"list_key_two": 2}],
        }
        assert normalized == expected
