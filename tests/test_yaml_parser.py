# tests/test_yaml_parser.py
import os
import unittest

import yaml  # For creating test files

from utils.common import (
    load_yaml_file,
    normalize_keys_recursive,
)  # Assuming yaml_parser is in PYTHONPATH


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
            f.write("Novel Concept: Title: Test Novel\nGenre: [Sci-Fi")  # Missing closing bracket

        self.empty_yaml_filepath = os.path.join(self.test_dir, "empty.yaml")
        with open(self.empty_yaml_filepath, "w", encoding="utf-8") as f:
            f.write("")  # Empty file

        self.non_dict_root_yaml_filepath = os.path.join(self.test_dir, "non_dict_root.yaml")
        with open(self.non_dict_root_yaml_filepath, "w", encoding="utf-8") as f:
            f.write("- item1\n- item2")  # Root is a list

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
        self.assertIsNotNone(data)
        # Explicit check for mypy
        assert data is not None
        self.assertIn("novel_concept", data)
        self.assertEqual(data["novel_concept"]["title"], "Test Novel")
        self.assertEqual(data["protagonist_traits"], ["Brave", "Smart"])

    def test_load_valid_yaml_raw_keys(self) -> None:
        data = load_yaml_file(self.valid_yaml_filepath, normalize_keys=False)
        self.assertIsNotNone(data)
        # Explicit check for mypy
        assert data is not None
        self.assertIn("Novel Concept", data)
        self.assertEqual(data["Novel Concept"]["Title"], "Test Novel")

    def test_load_non_existent_file(self) -> None:
        data = load_yaml_file("non_existent.yaml")
        self.assertIsNone(data)

    def test_load_malformed_yaml(self) -> None:
        data = load_yaml_file(self.malformed_yaml_filepath)
        self.assertIsNone(data)

    def test_load_empty_yaml(self) -> None:
        data = load_yaml_file(self.empty_yaml_filepath)
        self.assertEqual(data, {})  # Expecting an empty dictionary for an empty file

    def test_load_non_dict_root_yaml(self) -> None:
        # Current implementation of load_yaml_file logs error and returns None if root is not dict
        data = load_yaml_file(self.non_dict_root_yaml_filepath)
        self.assertIsNone(data)

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
        self.assertEqual(normalized, expected)


if __name__ == "__main__":
    unittest.main()
