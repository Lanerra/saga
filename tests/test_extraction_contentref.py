"""Test that extraction nodes consistently use ContentRefs."""

from unittest.mock import MagicMock, patch

import pytest

from core.langgraph.content_manager import ContentManager, ContentRef
from core.langgraph.nodes.extraction_nodes import consolidate_extraction
from core.langgraph.nodes.scene_extraction import extract_from_scenes
from core.langgraph.state import NarrativeState


@pytest.mark.asyncio
async def test_extract_from_scenes_externalizes_immediately():
    """Test that extract_from_scenes externalizes data immediately using ContentRefs."""
    # Setup mock state with scene drafts ref (not scenes dict)
    state: NarrativeState = {
        "current_chapter": 1,
        "project_dir": "/tmp/test_project",
        "title": "Test Novel",
        "genre": "Fantasy",
        "protagonist_name": "Alice",
    }

    # Mock the content manager and external dependencies
    with patch("core.langgraph.nodes.scene_extraction.ContentManager") as mock_cm_class:
        mock_cm = MagicMock(spec=ContentManager)
        mock_cm_class.return_value = mock_cm

        # Mock get_latest_version to return 0 (first version)
        mock_cm.get_latest_version.return_value = 0

        # Mock save_json to return ContentRef objects
        mock_entities_ref = ContentRef(
            path="/tmp/test_project/content/extracted_entities/chapter_1/v1.json",
            content_type="extracted_entities",
            version=1,
            size_bytes=1024,
        )
        mock_rels_ref = ContentRef(
            path="/tmp/test_project/content/extracted_relationships/chapter_1/v1.json",
            content_type="extracted_relationships",
            version=1,
            size_bytes=512,
        )
        mock_cm.save_json.return_value = mock_entities_ref  # First call returns entities ref
        mock_cm.save_json.side_effect = [
            mock_entities_ref,  # First call for entities
            mock_rels_ref,  # Second call for relationships
        ]

        # Mock get_scene_drafts to return scene texts
        with patch("core.langgraph.nodes.scene_extraction.get_scene_drafts") as mock_get_drafts:
            mock_get_drafts.return_value = ["Test scene 1", "Test scene 2"]

            # Mock extract_from_scene to return consolidated extraction results
            with patch("core.langgraph.nodes.scene_extraction.extract_from_scene") as mock_extract:
                # Setup mock extraction results (what extract_from_scene returns)
                mock_extract.return_value = {
                    "characters": [{"name": "Alice", "description": "Protagonist"}],
                    "world_items": [{"name": "Castle", "type": "Location"}],
                    "relationships": [{"source": "Alice", "target": "Bob", "type": "friendship"}],
                }

                # Call the function
                result = await extract_from_scenes(state)

                # Verify results
                assert "extracted_entities" in result
                assert result["extracted_entities"] == {}  # Should be cleared

                assert "extracted_relationships" in result
                assert result["extracted_relationships"] == []  # Should be cleared

                assert "extracted_entities_ref" in result
                assert result["extracted_entities_ref"] == mock_entities_ref

                assert "extracted_relationships_ref" in result
                assert result["extracted_relationships_ref"] == mock_rels_ref

                # Verify save_json was called twice (once for entities, once for relationships)
                assert mock_cm.save_json.call_count == 2


def test_consolidate_extraction_with_pre_externalized_data():
    """Test that consolidate_extraction handles pre-externalized data correctly."""
    # Setup state with pre-externalized data
    state: NarrativeState = {
        "current_chapter": 1,
        "project_dir": "/tmp/test_project",
        "extracted_entities_ref": ContentRef(
            path="/tmp/test_project/content/extracted_entities/chapter_1/v1.json",
            content_type="extracted_entities",
            version=1,
            size_bytes=1024,
        ),
        "extracted_relationships_ref": ContentRef(
            path="/tmp/test_project/content/extracted_relationships/chapter_1/v1.json",
            content_type="extracted_relationships",
            version=1,
            size_bytes=512,
        ),
    }

    # Mock the content manager
    with patch("core.langgraph.nodes.extraction_nodes.ContentManager") as mock_cm_class:
        mock_cm = MagicMock(spec=ContentManager)
        mock_cm_class.return_value = mock_cm

        # Mock exists to return True (files exist)
        mock_cm.exists.return_value = True

        # Call the function
        result = consolidate_extraction(state)

        # Verify results - should pass through the refs unchanged
        assert "extracted_entities" in result
        assert result["extracted_entities"] == {}  # Should be cleared

        assert "extracted_relationships" in result
        assert result["extracted_relationships"] == []  # Should be cleared

        assert "extracted_entities_ref" in result
        assert result["extracted_entities_ref"] == state["extracted_entities_ref"]

        assert "extracted_relationships_ref" in result
        assert result["extracted_relationships_ref"] == state["extracted_relationships_ref"]

        # Verify exists was called for both refs
        assert mock_cm.exists.call_count == 2


def test_consolidate_extraction_with_in_memory_data():
    """Test that consolidate_extraction still works with in-memory data (backward compatibility)."""
    # Setup state with in-memory data (no refs)
    state: NarrativeState = {
        "current_chapter": 1,
        "project_dir": "/tmp/test_project",
        "extracted_entities": {
            "characters": [{"name": "Alice", "description": "Protagonist"}],
            "world_items": [{"name": "Castle", "type": "Location"}],
        },
        "extracted_relationships": [{"source": "Alice", "target": "Bob", "type": "friendship"}],
    }

    # Mock the content manager and helper functions
    with patch("core.langgraph.nodes.extraction_nodes.ContentManager") as mock_cm_class:
        mock_cm = MagicMock(spec=ContentManager)
        mock_cm_class.return_value = mock_cm

        # Mock get_latest_version to return 0 (first version)
        mock_cm.get_latest_version.return_value = 0

        # Mock the helper functions from content_manager module
        with patch("core.langgraph.content_manager.get_extracted_entities") as mock_get_entities:
            with patch("core.langgraph.content_manager.get_extracted_relationships") as mock_get_rels:
                with patch("core.langgraph.content_manager.save_extracted_entities") as mock_save_entities:
                    with patch("core.langgraph.content_manager.save_extracted_relationships") as mock_save_rels:
                        # Setup mock returns
                        mock_get_entities.return_value = state["extracted_entities"]
                        mock_get_rels.return_value = state["extracted_relationships"]

                        mock_entities_ref = ContentRef(
                            path="/tmp/test_project/content/extracted_entities/chapter_1/v1.json",
                            content_type="extracted_entities",
                            version=1,
                            size_bytes=1024,
                        )
                        mock_rels_ref = ContentRef(
                            path="/tmp/test_project/content/extracted_relationships/chapter_1/v1.json",
                            content_type="extracted_relationships",
                            version=1,
                            size_bytes=512,
                        )
                        mock_save_entities.return_value = mock_entities_ref
                        mock_save_rels.return_value = mock_rels_ref

                        # Call the function
                        result = consolidate_extraction(state)

                        # Verify results
                        assert "extracted_entities" in result
                        assert result["extracted_entities"] == {}  # Should be cleared

                        assert "extracted_relationships" in result
                        assert result["extracted_relationships"] == []  # Should be cleared

                        assert "extracted_entities_ref" in result
                        assert result["extracted_entities_ref"] == mock_entities_ref

                        assert "extracted_relationships_ref" in result
                        assert result["extracted_relationships_ref"] == mock_rels_ref

                        # Verify helper functions were called (3 times for entities: lines 95, 96, 102)
                        assert mock_get_entities.call_count == 3
                        assert mock_get_rels.call_count == 2  # Lines 97 and 127
                        mock_save_entities.assert_called_once()
                        mock_save_rels.assert_called_once()
