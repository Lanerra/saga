# tests/test_yaml_parser.py
import os
import unittest

import pytest
import yaml

from utils.common import (
    load_yaml_file,
    normalize_keys_recursive,
)


class TestYamlParsing(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = "temp_test_yaml_files"
        os.makedirs(self.test_dir, exist_ok=True)

        self.valid_yaml_content = {
            "Novel Concept": {"Title": "Test Novel", "Genre": "Sci-Fi"},
            "protagonist_traits": ["Brave", "Smart"],
        }
        self.valid_yaml_filepath = os.path.join(self.test_dir, "valid.yaml")
        with open(self.valid_yaml_filepath, "w", encoding="utf-8") as f:
            yaml.dump(self.valid_yaml_content, f)

        self.malformed_yaml_filepath = os.path.join(self.test_dir, "malformed.yaml")
        with open(self.malformed_yaml_filepath, "w", encoding="utf-8") as f:
            f.write("Novel Concept: Title: Test Novel\nGenre: [Sci-Fi")

        self.empty_yaml_filepath = os.path.join(self.test_dir, "empty.yaml")
        with open(self.empty_yaml_filepath, "w", encoding="utf-8") as f:
            f.write("")

        self.non_dict_root_yaml_filepath = os.path.join(self.test_dir, "non_dict_root.yaml")
        with open(self.non_dict_root_yaml_filepath, "w", encoding="utf-8") as f:
            f.write("- item1\n- item2")

    def tearDown(self) -> None:
        if os.path.exists(self.valid_yaml_filepath):
            os.remove(self.valid_yaml_filepath)
        if os.path.exists(self.malformed_yaml_filepath):
            os.remove(self.malformed_yaml_filepath)
        if os.path.exists(self.empty_yaml_filepath):
            os.remove(self.empty_yaml_filepath)
        if os.path.exists(self.non_dict_root_yaml_filepath):
            os.remove(self.non_dict_root_yaml_filepath)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)

    def test_load_valid_yaml_normalized_keys(self) -> None:
        data = load_yaml_file(self.valid_yaml_filepath, normalize_keys=True)
        assert "novel_concept" in data
        assert data["novel_concept"]["title"] == "Test Novel"
        assert data["protagonist_traits"] == ["Brave", "Smart"]

    def test_load_valid_yaml_raw_keys(self) -> None:
        data = load_yaml_file(self.valid_yaml_filepath, normalize_keys=False)
        assert "Novel Concept" in data
        assert data["Novel Concept"]["Title"] == "Test Novel"

    def test_load_non_existent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_yaml_file("non_existent.yaml")

    def test_load_malformed_yaml_raises(self) -> None:
        with pytest.raises(yaml.YAMLError):
            load_yaml_file(self.malformed_yaml_filepath)

    def test_load_empty_yaml(self) -> None:
        data = load_yaml_file(self.empty_yaml_filepath)
        assert data == {}

    def test_load_non_dict_root_yaml_raises(self) -> None:
        with pytest.raises(ValueError, match="must have a dictionary as its root element"):
            load_yaml_file(self.non_dict_root_yaml_filepath)

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


if __name__ == "__main__":
    unittest.main()
