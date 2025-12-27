# tests/core/langgraph/test_content_manager_strict_reads.py
from __future__ import annotations

from pathlib import Path

import pytest

from core.exceptions import ContentIntegrityError
from core.langgraph.content_manager import ContentManager


def test_load_text_strict_succeeds_when_checksum_and_size_match(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))

    ref = manager.save_text("hello", content_type="unit_test_text", identifier="sample", version=1)

    loaded = manager.load_text_strict(ref)
    assert loaded == "hello"


def test_load_text_strict_raises_when_file_is_modified_after_ref_creation(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))

    ref = manager.save_text("hello", content_type="unit_test_text", identifier="sample", version=1)

    full_path = tmp_path / ref["path"]
    assert full_path.is_file()

    full_path.write_text("corrupted", encoding="utf-8")

    with pytest.raises(ContentIntegrityError) as exc:
        manager.load_text_strict(ref)

    assert "strict read detected size mismatch" in str(exc.value)


def test_load_text_strict_raises_when_metadata_is_missing(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))

    ref = manager.save_text("hello", content_type="unit_test_text", identifier="sample", version=1)

    ref_without_integrity_metadata = {
        "path": ref["path"],
        "content_type": ref["content_type"],
        "version": ref["version"],
    }

    with pytest.raises(ContentIntegrityError) as exc:
        manager.load_text_strict(ref_without_integrity_metadata)  # type: ignore[arg-type]

    assert "strict read requires ContentRef.size_bytes metadata" in str(exc.value)


def test_load_json_strict_succeeds_when_checksum_and_size_match(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))

    payload = {"a": 1, "b": ["x", "y"]}
    ref = manager.save_json(payload, content_type="unit_test_json", identifier="sample", version=1)

    loaded = manager.load_json_strict(ref)
    assert loaded == payload


def test_load_json_strict_raises_when_file_is_modified_after_ref_creation(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))

    payload = {"a": 1}
    ref = manager.save_json(payload, content_type="unit_test_json", identifier="sample", version=1)

    full_path = tmp_path / ref["path"]
    assert full_path.is_file()

    full_path.write_text('{"a": 2}', encoding="utf-8")

    with pytest.raises(ContentIntegrityError) as exc:
        manager.load_json_strict(ref)

    assert "strict read detected size mismatch" in str(exc.value)