# tests/test_langgraph/test_content_manager_embeddings.py
from __future__ import annotations

from pathlib import Path

import pytest

from core.langgraph.content_manager import ContentManager, load_embedding, save_embedding


def test_embedding_round_trip_uses_safe_json_format(tmp_path: Path) -> None:
    """
    Embeddings must round-trip without pickle.

    This test asserts:
    - [`save_embedding()`](core/langgraph/content_manager.py:513) produces a `.json` artifact
      (safe, human-inspectable format).
    - [`load_embedding()`](core/langgraph/content_manager.py:523) returns the same numeric vector.
    """
    manager = ContentManager(str(tmp_path))

    embedding = [0.1, 0.2, 3.0]
    ref = save_embedding(manager, embedding, chapter=1, version=1)

    # Stored under `.saga/content/embedding/...` and must be a safe extension
    assert ref["path"].endswith(".json")

    full_path = tmp_path / ref["path"]
    assert full_path.is_file()

    loaded = load_embedding(manager, ref)
    assert loaded == embedding


def test_load_binary_refuses_legacy_pickle_by_default(tmp_path: Path) -> None:
    """
    Default behavior must be safe: `.pkl` artifacts are refused.

    We create a fake legacy pickle file and ensure
    [`ContentManager.load_binary()`](core/langgraph/content_manager.py:284) raises.
    """
    manager = ContentManager(str(tmp_path))

    legacy_path = tmp_path / ".saga" / "content" / "embedding" / "chapter_1_v1.pkl"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_bytes(b"not a pickle")

    with pytest.raises(ValueError) as exc:
        manager.load_binary(str(legacy_path.relative_to(tmp_path)))

    assert "Refusing to load legacy pickle artifact" in str(exc.value)
