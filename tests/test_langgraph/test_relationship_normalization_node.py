import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.langgraph.nodes.relationship_normalization_node import normalize_relationships
from core.langgraph.state import NarrativeState


@pytest.mark.asyncio
async def test_normalize_relationships_node(tmp_path: Path):
    """
    Test the normalize_relationships node integration.
    """
    project_dir = str(tmp_path)

    # Create extraction file
    extraction_dir = tmp_path / "extraction"
    extraction_dir.mkdir()

    # Mock extract of relationships
    rels = [
        {
            "source_name": "Alice",
            "target_name": "Bob",
            "relationship_type": "FRIENDS_WITH",
            "description": "Close friends",
            "chapter": 1,
            "confidence": 0.9,
        },
        {
            "source_name": "Bob",
            "target_name": "Charlie",
            "relationship_type": "works_with",  # Small case
            "description": "Colleagues",
            "chapter": 1,
            "confidence": 0.8,
        },
    ]

    # Initialize vocabulary with canonical "WORKS_WITH"
    vocabulary = {
        "WORKS_WITH": {
            "canonical_type": "WORKS_WITH",
            "usage_count": 5,
            "first_used_chapter": 0,
            "example_descriptions": [],
            "synonyms": [],
            "last_used_chapter": 0,
        }
    }

    # State with extracted relationships (usually passed as objects or dicts via content manager?
    # normalize_relationships calls get_extracted_relationships(state, content_manager))
    # content_manager reads from disk.

    # We need to write the file that content_manager reads.
    # ContentManager.get_extracted_relationships reads {chapter}_relationships.json

    rel_file = extraction_dir / "chapter_001_relationships.json"
    with open(rel_file, "w") as f:
        json.dump(rels, f)

    state: NarrativeState = {
        "project_dir": project_dir,
        "current_chapter": 1,
        "relationship_vocabulary": vocabulary,
        "extracted_relationships_ref": str(rel_file),  # Assuming ref pattern
    }

    # We need to mock get_extracted_relationships if it relies on complex paths or just ensure the file exists where it expects.
    # Looking at content_manager.get_extracted_relationships implementation (not shown fully), it likely looks up by chapter.

    # Let's mock ContentManager.get_extracted_relationships to be safe and simple
    with patch(
        "core.langgraph.nodes.relationship_normalization_node.get_extracted_relationships"
    ) as mock_get:
        mock_get.return_value = rels

        # Also mock set_extracted_relationships to verify output
        with patch(
            "core.langgraph.nodes.relationship_normalization_node.set_extracted_relationships"
        ) as mock_set:
            # Run node
            new_state = await normalize_relationships(state)

            assert new_state["current_node"] == "normalize_relationships"

            # Verify vocabulary update
            new_vocab = new_state["relationship_vocabulary"]
            assert "FRIENDS_WITH" in new_vocab  # Novel relationship added
            assert new_vocab["WORKS_WITH"]["usage_count"] == 6  # Incremented

            # Verify normalized relationships
            # Check arguments to set_extracted_relationships
            args, _ = mock_set.call_args
            # args[0] is content_manager, args[1] is normalized_rels, args[2] is state
            normalized_rels = args[1]

            assert len(normalized_rels) == 2
            assert normalized_rels[0].relationship_type == "FRIENDS_WITH"
            assert (
                normalized_rels[1].relationship_type == "WORKS_WITH"
            )  # Normalized from works_with

            # Verify stats in state
            assert (
                new_state["relationships_normalized_this_chapter"] == 1
            )  # works_with -> WORKS_WITH
            assert new_state["relationships_novel_this_chapter"] == 1  # FRIENDS_WITH
