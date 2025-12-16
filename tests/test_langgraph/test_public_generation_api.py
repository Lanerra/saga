# tests/test_langgraph/test_public_generation_api.py
from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from core.langgraph import generate_chapter, generate_chapter_single_shot
from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.embedding_node import generate_embedding
from core.langgraph.nodes.finalize_node import finalize_chapter


def test_generate_chapter_export_is_scene_based_subgraph() -> None:
    """
    Ensure the public export `core.langgraph.generate_chapter` is the canonical
    scene-based generation API surface (not the legacy single-shot async node).
    """
    assert not inspect.iscoroutinefunction(generate_chapter), "generate_chapter should be a subgraph factory, not an async node function"
    assert inspect.iscoroutinefunction(generate_chapter_single_shot), "generate_chapter_single_shot should remain an async node function for backcompat"

    graph = generate_chapter()

    # We don't assert exact LangGraph internal types here; just contract that it's
    # an executable graph object.
    assert hasattr(graph, "ainvoke"), "generate_chapter() should return a compiled graph with .ainvoke()"
    assert hasattr(graph, "astream"), "generate_chapter() should return a compiled graph with .astream()"


@pytest.mark.asyncio
async def test_embeddings_computed_once_embedding_node_then_finalize_reuses_ref(tmp_path: Path) -> None:
    """
    Ensure embeddings are computed once when the embedding node runs, and
    finalize reuses `embedding_ref` rather than recomputing.

    Contract:
    - [`generate_embedding()`](core/langgraph/nodes/embedding_node.py:19) computes embedding and writes `embedding_ref`.
    - [`finalize_chapter()`](core/langgraph/nodes/finalize_node.py:33) reuses `embedding_ref` (no extra embedding call).
    """
    project_dir = str(tmp_path / "proj")
    manager = ContentManager(project_dir)

    # Provide draft for embedding and finalize.
    draft_ref = manager.save_text("Hello chapter draft.", "draft", "chapter_1", version=1)

    state = {
        "project_dir": project_dir,
        "current_chapter": 1,
        "draft_ref": draft_ref,
        "draft_word_count": 3,
        # Provide a summary list so finalize can load one (optional but keeps the path exercised).
        "summaries_ref": manager.save_list_of_texts(["Summary."], "summaries", "all", version=1),
    }

    fake_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    # Patch the shared llm_service object once so both nodes see the same mocked method.
    with patch(
        "core.llm_interface_refactored.llm_service.async_get_embedding",
        new=AsyncMock(return_value=fake_embedding),
    ) as mock_get_embedding:
        state_with_embedding = await generate_embedding(state)  # type: ignore[arg-type]
        assert state_with_embedding.get("embedding_ref"), "embedding node must set embedding_ref"

        with patch(
            "core.langgraph.nodes.finalize_node.save_chapter_data_to_db",
            new=AsyncMock(return_value=None),
        ) as mock_save:
            await finalize_chapter(state_with_embedding)  # type: ignore[arg-type]

        # Exactly once overall: embedding computed in embedding node, NOT recomputed in finalize.
        assert mock_get_embedding.call_count == 1

        # Finalize must persist the embedding (as numpy array) to Neo4j save call.
        assert mock_save.call_count == 1
        saved_embedding = mock_save.call_args.kwargs["embedding_array"]
        assert isinstance(saved_embedding, np.ndarray)
        assert saved_embedding.tolist() == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)
