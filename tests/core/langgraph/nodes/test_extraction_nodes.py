"""Tests for consolidate_extraction node with real ContentManager."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.langgraph.content_manager import (
    ContentManager,
    save_extracted_entities,
    save_extracted_relationships,
)
from core.langgraph.nodes.extraction_nodes import consolidate_extraction

EXAMPLE_ENTITIES = {
    "characters": [
        {
            "name": "Alice",
            "type": "Character",
            "description": "The protagonist",
            "first_appearance_chapter": 1,
        }
    ],
    "world_items": [
        {
            "name": "Crystal Tower",
            "type": "Place",
            "description": "A tall tower",
            "first_appearance_chapter": 1,
        }
    ],
}

EXAMPLE_RELATIONSHIPS = [
    {
        "source_name": "Alice",
        "target_name": "Crystal Tower",
        "relationship_type": "VISITED",
        "description": "Alice visited the tower",
        "chapter": 1,
        "confidence": 0.9,
    }
]


class TestConsolidateExtractionPreExternalized:
    """Pre-externalized path: refs already exist on disk."""

    @pytest.mark.asyncio
    async def test_returns_existing_refs_unchanged(self, tmp_path: Path) -> None:
        content_manager = ContentManager(str(tmp_path))

        entities_ref = save_extracted_entities(content_manager, EXAMPLE_ENTITIES, chapter=1, version=1)
        relationships_ref = save_extracted_relationships(content_manager, EXAMPLE_RELATIONSHIPS, chapter=1, version=1)

        state = {
            "project_dir": str(tmp_path),
            "current_chapter": 1,
            "extracted_entities_ref": entities_ref,
            "extracted_relationships_ref": relationships_ref,
        }

        result = await consolidate_extraction(state)

        assert result["extracted_entities_ref"] == entities_ref
        assert result["extracted_relationships_ref"] == relationships_ref
        assert result["current_node"] == "consolidate_extraction"

    @pytest.mark.asyncio
    async def test_entities_file_missing_returns_fatal_error(self, tmp_path: Path) -> None:
        content_manager = ContentManager(str(tmp_path))

        entities_ref = save_extracted_entities(content_manager, EXAMPLE_ENTITIES, chapter=1, version=1)
        relationships_ref = save_extracted_relationships(content_manager, EXAMPLE_RELATIONSHIPS, chapter=1, version=1)

        entities_path = tmp_path / entities_ref["path"]
        entities_path.unlink()

        state = {
            "project_dir": str(tmp_path),
            "current_chapter": 1,
            "extracted_entities_ref": entities_ref,
            "extracted_relationships_ref": relationships_ref,
        }

        result = await consolidate_extraction(state)

        assert result["has_fatal_error"] is True
        assert result["error_node"] == "consolidate_extraction"
        assert "not found" in result["last_error"]

    @pytest.mark.asyncio
    async def test_relationships_file_missing_returns_fatal_error(self, tmp_path: Path) -> None:
        content_manager = ContentManager(str(tmp_path))

        entities_ref = save_extracted_entities(content_manager, EXAMPLE_ENTITIES, chapter=1, version=1)
        relationships_ref = save_extracted_relationships(content_manager, EXAMPLE_RELATIONSHIPS, chapter=1, version=1)

        relationships_path = tmp_path / relationships_ref["path"]
        relationships_path.unlink()

        state = {
            "project_dir": str(tmp_path),
            "current_chapter": 1,
            "extracted_entities_ref": entities_ref,
            "extracted_relationships_ref": relationships_ref,
        }

        result = await consolidate_extraction(state)

        assert result["has_fatal_error"] is True
        assert result["error_node"] == "consolidate_extraction"
        assert "not found" in result["last_error"]


class TestConsolidateExtractionBackwardCompatibility:
    """Backward-compatibility path: in-memory data gets externalized."""

    @pytest.mark.asyncio
    async def test_externalizes_in_memory_data(self, tmp_path: Path) -> None:
        state = {
            "project_dir": str(tmp_path),
            "current_chapter": 2,
            "extracted_entities_ref": None,
            "extracted_relationships_ref": None,
            "extracted_entities": EXAMPLE_ENTITIES,
            "extracted_relationships": EXAMPLE_RELATIONSHIPS,
        }

        result = await consolidate_extraction(state)

        assert result["current_node"] == "consolidate_extraction"

        new_entities_ref = result["extracted_entities_ref"]
        new_relationships_ref = result["extracted_relationships_ref"]

        assert new_entities_ref is not None
        assert new_relationships_ref is not None
        assert new_entities_ref["content_type"] == "extracted_entities"
        assert new_relationships_ref["content_type"] == "extracted_relationships"
        assert new_entities_ref["size_bytes"] > 0
        assert new_relationships_ref["size_bytes"] > 0

        content_manager = ContentManager(str(tmp_path))
        assert content_manager.exists(new_entities_ref) is True
        assert content_manager.exists(new_relationships_ref) is True

    @pytest.mark.asyncio
    async def test_externalizes_empty_data(self, tmp_path: Path) -> None:
        state = {
            "project_dir": str(tmp_path),
            "current_chapter": 1,
            "extracted_entities_ref": None,
            "extracted_relationships_ref": None,
            "extracted_entities": {},
            "extracted_relationships": [],
        }

        result = await consolidate_extraction(state)

        assert result["current_node"] == "consolidate_extraction"
        assert result["extracted_entities_ref"] is not None
        assert result["extracted_relationships_ref"] is not None

        content_manager = ContentManager(str(tmp_path))
        assert content_manager.exists(result["extracted_entities_ref"]) is True
        assert content_manager.exists(result["extracted_relationships_ref"]) is True
