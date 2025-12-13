# tests/test_langgraph/test_embedding_node_safe_embeddings.py
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from core.langgraph.content_manager import ContentManager, load_embedding
from core.langgraph.nodes.embedding_node import generate_embedding


@pytest.mark.asyncio
async def test_generate_embedding_externalizes_as_json_and_loads(tmp_path: Path) -> None:
    """
    Verify the embedding workflow path works without pickle:

    - [`generate_embedding()`](core/langgraph/nodes/embedding_node.py:19) writes an externalized embedding artifact.
    - The artifact is stored in a safe format (`.json`) via
      [`ContentManager.save_binary()`](core/langgraph/content_manager.py:189).
    - The saved embedding can be read back via
      [`load_embedding()`](core/langgraph/content_manager.py:523).
    """
    project_dir = str(tmp_path)
    manager = ContentManager(project_dir)

    # Provide externalized draft text required by get_draft_text()
    draft_ref = manager.save_text("Hello chapter draft.", "draft", "chapter_1", version=1)

    state = {
        "project_dir": project_dir,
        "current_chapter": 1,
        "draft_ref": draft_ref,
    }

    fake_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    with patch(
        "core.langgraph.nodes.embedding_node.llm_service.async_get_embedding",
        new=AsyncMock(return_value=fake_embedding),
    ):
        new_state = await generate_embedding(state)  # type: ignore[arg-type]

    assert new_state["current_node"] == "generate_embedding"
    assert new_state.get("embedding_ref"), "generate_embedding must set embedding_ref on success"

    embedding_ref = new_state["embedding_ref"]
    assert isinstance(embedding_ref, dict)
    assert embedding_ref["path"].endswith(".json"), "Embeddings must not be stored as pickle by default"

    # Ensure we can load the embedding back safely
    loaded = load_embedding(manager, embedding_ref)

    # The mocked embedding is float32; round-trip through JSON may reflect float32->float conversions.
    assert loaded == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)
