from __future__ import annotations

from pathlib import Path

import yaml

from utils.file_io import write_text_file, write_yaml_file


def test_write_text_file_creates_parent_and_writes_utf8(tmp_path: Path) -> None:
    # Arrange
    target = tmp_path / "nested" / "file.txt"

    # Act
    write_text_file(target, "héllo")

    # Assert
    assert target.parent.is_dir(), "Parent directory was not created"
    assert target.is_file(), "File was not created"

    content = target.read_text(encoding="utf-8")
    assert content == "héllo"


def test_write_yaml_file_creates_parent_and_writes_mapping(tmp_path: Path) -> None:
    # Arrange
    target = tmp_path / "nested" / "data.yaml"
    payload = {"key": "value", "num": 1}

    # Act
    write_yaml_file(target, payload)

    # Assert structure
    assert target.is_file(), "YAML file was not created"

    loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert loaded == payload

    # Assert line endings are normalized to LF (no CRLF)
    raw = target.read_bytes().decode("utf-8")
    assert "\r\n" not in raw, "YAML output should use LF newlines only"
