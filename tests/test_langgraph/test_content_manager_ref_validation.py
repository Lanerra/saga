# tests/test_langgraph/test_content_manager_ref_validation.py
from __future__ import annotations

from pathlib import Path

import pytest

from core.langgraph.content_manager import ContentManager


def test_load_text_rejects_partial_content_ref_missing_path(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))

    with pytest.raises(ValueError) as exc:
        manager.load_text({"content_type": "draft"})  # missing required "path"

    # Must be explicit/actionable (and not a raw KeyError)
    assert "required key 'path'" in str(exc.value)
    assert "ContentManager.load_text" in str(exc.value)


def test_load_json_rejects_partial_content_ref_missing_path(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))

    with pytest.raises(ValueError) as exc:
        manager.load_json({"content_type": "unit_test_json"})  # missing required "path"

    assert "required key 'path'" in str(exc.value)
    assert "ContentManager.load_json" in str(exc.value)


def test_load_binary_rejects_partial_content_ref_missing_path(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))

    with pytest.raises(ValueError) as exc:
        manager.load_binary({"content_type": "embedding"})  # missing required "path"

    assert "required key 'path'" in str(exc.value)
    assert "ContentManager.load_binary" in str(exc.value)


def test_loaders_still_accept_valid_content_refs_and_paths(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))

    # Text round-trip via ContentRef
    text_ref = manager.save_text("hello", content_type="unit_test_text", identifier="sample", version=1)
    assert manager.load_text(text_ref) == "hello"

    # JSON round-trip via ContentRef
    json_ref = manager.save_json({"a": 1}, content_type="unit_test_json", identifier="sample", version=1)
    assert manager.load_json(json_ref) == {"a": 1}

    # Binary round-trip via ContentRef
    binary_ref = manager.save_binary(b"abc", content_type="unit_test_bin", identifier="sample", version=1)
    assert manager.load_binary(binary_ref) == b"abc"

    # Also accept Path objects (consistency with ContentRef | str | Path loader contract)
    assert manager.load_text(Path(text_ref["path"])) == "hello"
    assert manager.load_json(Path(json_ref["path"])) == {"a": 1}
    assert manager.load_binary(Path(binary_ref["path"])) == b"abc"
