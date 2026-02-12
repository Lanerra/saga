# tests/test_langgraph/test_file_io.py
from __future__ import annotations

from pathlib import Path

import yaml

from utils.file_io import write_text_file, write_yaml_file


def test_write_text_file_creates_parent_and_writes_utf8(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "file.txt"
    write_text_file(target, "héllo")

    assert target.parent.is_dir()
    assert target.is_file()
    assert target.read_text(encoding="utf-8") == "héllo"


def test_write_yaml_file_creates_parent_and_writes_mapping(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "data.yaml"
    payload = {"key": "value", "num": 1}
    write_yaml_file(target, payload)

    assert target.is_file()
    loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert loaded == payload

    raw = target.read_bytes().decode("utf-8")
    assert "\r\n" not in raw


def test_atomic_write_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "atomic.txt"
    write_text_file(target, "content")

    siblings = list(tmp_path.iterdir())
    assert siblings == [target]


def test_atomic_write_overwrites_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "overwrite.txt"
    write_text_file(target, "original")
    write_text_file(target, "replaced")

    assert target.read_text(encoding="utf-8") == "replaced"
