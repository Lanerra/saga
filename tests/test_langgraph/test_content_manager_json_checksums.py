# tests/test_langgraph/test_content_manager_json_checksums.py
from __future__ import annotations

import hashlib
from pathlib import Path

from core.langgraph.content_manager import ContentManager


def test_save_json_checksum_matches_file_bytes(tmp_path: Path) -> None:
    """
    Regression test for LANGGRAPH-012.

    [`ContentManager.save_json()`](core/langgraph/content_manager.py:141) must compute its
    `checksum` from the exact bytes written to disk (not from a different serialization).
    """
    manager = ContentManager(str(tmp_path))

    # Include a non-ASCII character to ensure ensure_ascii=False behavior is covered.
    payload = {
        "greeting": "héllo",
        "numbers": [1, 2, 3],
        "nested": {"a": 1, "b": True},
    }

    ref = manager.save_json(payload, content_type="unit_test_json", identifier="sample", version=1)

    full_path = tmp_path / ref["path"]
    assert full_path.is_file()

    file_bytes = full_path.read_bytes()

    # Checksum must match the exact bytes on disk.
    expected_checksum = hashlib.sha256(file_bytes).hexdigest()
    assert ref["checksum"] == expected_checksum

    # Metadata should match the stored artifact, too.
    assert ref["size_bytes"] == len(file_bytes)

    # Sanity: content is valid JSON and round-trips correctly.
    loaded = manager.load_json(ref)
    assert loaded == payload

    # The written JSON should preserve unicode characters (ensure_ascii=False),
    # i.e. it should not contain \u-escaped versions of "é".
    file_text = file_bytes.decode("utf-8")
    assert "héllo" in file_text
    assert "\\u00e9" not in file_text
