# tests/test_langgraph/test_relationship_normalization_node.py
from pathlib import Path
from unittest.mock import patch

import pytest

from core.langgraph.nodes.relationship_normalization_node import normalize_relationships
from core.langgraph.state import NarrativeState


@pytest.mark.asyncio
async def test_normalize_relationships_node_updates_ref(tmp_path: Path):
    """
    Unit test for [`normalize_relationships()`](core/langgraph/nodes/relationship_normalization_node.py:32).

    Scope:
    - Avoid embedding/LLM calls by only testing case/punctuation normalization against an existing vocabulary key.
    - Ensure the node returns an updated `extracted_relationships_ref`, so downstream nodes that prefer the ref
      (e.g. commit via [`get_extracted_relationships()`](core/langgraph/content_manager.py:929)) see normalized data.
    - Ensure the in-memory mirror list is cleared to prevent checkpoint bloat.
    """
    project_dir = str(tmp_path)

    # Normalization is deprecated (disabled by default) but this test verifies
    # the node logic when explicitly enabled.

    # Only include relationship types that will normalize via exact canonicalization,
    # so we don't invoke embedding similarity (and therefore don't require LLM mocks).
    rels = [
        {
            "source_name": "Bob",
            "target_name": "Charlie",
            "relationship_type": "allies_with",  # case variant of ALLIES_WITH
            "description": "Colleagues",
            "chapter": 1,
            "confidence": 0.8,
        },
        {
            "source_name": "Alice",
            "target_name": "Bob",
            "relationship_type": "ALLIES_WITH",  # already canonical
            "description": "Work partners",
            "chapter": 1,
            "confidence": 0.9,
        },
    ]

    vocabulary = {
        "ALLIES_WITH": {
            "canonical_type": "ALLIES_WITH",
            "usage_count": 5,
            "first_used_chapter": 0,
            "example_descriptions": [],
            "synonyms": [],
            "last_used_chapter": 0,
        }
    }

    state: NarrativeState = {
        "project_dir": project_dir,
        "current_chapter": 1,
        "relationship_vocabulary": vocabulary,
        # doesn't matter for this unit test because we mock the getter,
        # but the node expects the field to possibly exist.
        "extracted_relationships_ref": {"path": "dummy.json", "version": 1, "content_type": "extracted_relationships"},
    }

    with patch("config.ENABLE_RELATIONSHIP_NORMALIZATION", True), patch("core.langgraph.nodes.relationship_normalization_node.get_extracted_relationships") as mock_get:
        mock_get.return_value = rels

        with patch("core.langgraph.nodes.relationship_normalization_node.set_extracted_relationships") as mock_set:
            normalized_ref = {"path": ".saga/content/extracted_relationships/chapter_1_v2.json", "version": 2}
            mock_set.return_value = normalized_ref

            new_state = await normalize_relationships(state)

            assert new_state["current_node"] == "normalize_relationships"

            # Node must return updated ref so downstream consumers read normalized content.
            assert new_state["extracted_relationships_ref"] == normalized_ref

            # In-memory mirror must be cleared after externalization.
            assert new_state["extracted_relationships"] == []

            # Verify normalized relationships were computed and persisted via setter call.
            args, _ = mock_set.call_args
            normalized_rels = args[1]

            assert len(normalized_rels) == 2
            assert normalized_rels[0].relationship_type == "ALLIES_WITH"
            assert normalized_rels[1].relationship_type == "ALLIES_WITH"

            assert new_state["relationships_normalized_this_chapter"] == 1
            assert new_state["relationships_novel_this_chapter"] == 0
