"""Simple integration tests for core/langgraph/nodes/commit_node.py."""

from unittest.mock import MagicMock, patch

import pytest

from core.langgraph.nodes.commit_node import commit_to_graph


class TestCommitNodeIntegration:
    """Integration tests for the commit node."""

    @pytest.mark.asyncio
    async def test_commit_to_graph_with_valid_state(self) -> None:
        """Test that commit_to_graph handles valid state without errors."""
        mock_state = {
            "chapter": 1,
            "project_dir": "/tmp/test_project",
            "extracted_entities_ref": {
                "path": ".saga/content/extracted_entities/chapter_1.json",
                "content_type": "extracted_entities",
                "version": 1,
                "size_bytes": 100,
                "checksum": "abc123",
            },
        }

        with patch("core.langgraph.nodes.commit_node.ContentManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm_class.return_value = mock_cm
            mock_cm.read_json.return_value = {
                "characters": [{"name": "Alice", "entity_type": "character"}],
                "world_items": [],
            }

            result = await commit_to_graph(mock_state)  # type: ignore[arg-type]

            # Should return a valid state dict
            assert isinstance(result, dict)

            # Should update current_node
            assert result["current_node"] == "commit_to_graph"

    @pytest.mark.asyncio
    async def test_commit_to_graph_with_empty_extractions(self) -> None:
        """Test that commit_to_graph handles empty extractions."""
        mock_state = {
            "chapter": 1,
            "project_dir": "/tmp/test_project",
            "extracted_entities_ref": {
                "path": ".saga/content/extracted_entities/chapter_1.json",
                "content_type": "extracted_entities",
                "version": 1,
                "size_bytes": 0,
                "checksum": "empty",
            },
        }

        with patch("core.langgraph.nodes.commit_node.ContentManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm_class.return_value = mock_cm
            mock_cm.read_json.return_value = {
                "characters": [],
                "world_items": [],
            }

            result = await commit_to_graph(mock_state)  # type: ignore[arg-type]

            # Should return a valid state dict
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_commit_to_graph_with_relationships(self) -> None:
        """Test that commit_to_graph handles relationships."""
        mock_state = {
            "chapter": 1,
            "project_dir": "/tmp/test_project",
            "extracted_entities_ref": {
                "path": ".saga/content/extracted_entities/chapter_1.json",
                "content_type": "extracted_entities",
                "version": 1,
                "size_bytes": 100,
                "checksum": "abc123",
            },
            "extracted_relationships_ref": {
                "path": ".saga/content/extracted_relationships/chapter_1.json",
                "content_type": "extracted_relationships",
                "version": 1,
                "size_bytes": 50,
                "checksum": "def456",
            },
        }

        with patch("core.langgraph.nodes.commit_node.ContentManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm_class.return_value = mock_cm
            mock_cm.read_json.side_effect = [
                {
                    "characters": [{"name": "Alice", "entity_type": "character"}],
                    "world_items": [],
                },
                [
                    {
                        "source_name": "Alice",
                        "target_name": "Bob",
                        "relationship_type": "knows",
                        "confidence": 0.9,
                    },
                ],
            ]

            result = await commit_to_graph(mock_state)  # type: ignore[arg-type]

            # Should return a valid state dict
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_commit_to_graph_handles_database_error(self) -> None:
        """Test that commit_to_graph handles database errors gracefully."""
        mock_state = {
            "chapter": 1,
            "project_dir": "/tmp/test_project",
            "extracted_entities_ref": {
                "path": ".saga/content/extracted_entities/chapter_1.json",
                "content_type": "extracted_entities",
                "version": 1,
                "size_bytes": 100,
                "checksum": "abc123",
            },
        }

        with patch("core.langgraph.nodes.commit_node.ContentManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm_class.return_value = mock_cm
            mock_cm.read_json.return_value = {
                "characters": [{"name": "Alice", "entity_type": "character"}],
                "world_items": [],
            }

            with patch("core.db_manager.neo4j_manager.execute_cypher_batch") as mock_execute:
                mock_execute.side_effect = Exception("Database error")

                result = await commit_to_graph(mock_state)  # type: ignore[arg-type]

                # Should return state with error information
                assert "has_fatal_error" in result
                assert result["has_fatal_error"] is True

    @pytest.mark.asyncio
    async def test_commit_to_graph_handles_missing_content(self) -> None:
        """Test that commit_to_graph handles missing content references."""
        mock_state = {
            "chapter": 1,
            "project_dir": "/tmp/test_project",
            # No extracted_entities_ref
        }

        result = await commit_to_graph(mock_state)  # type: ignore[arg-type]

        # Should return state with error information
        assert "has_fatal_error" in result
        assert result["has_fatal_error"] is True

    @pytest.mark.asyncio
    async def test_commit_to_graph_preserves_state(self) -> None:
        """Test that commit_to_graph preserves existing state fields."""
        mock_state = {
            "chapter": 1,
            "project_dir": "/tmp/test_project",
            "some_existing_field": "preserve_this",
            "draft_ref": {
                "path": ".saga/content/drafts/chapter_1.json",
                "content_type": "draft",
                "version": 1,
                "size_bytes": 100,
                "checksum": "draft123",
            },
            "extracted_entities_ref": {
                "path": ".saga/content/extracted_entities/chapter_1.json",
                "content_type": "extracted_entities",
                "version": 1,
                "size_bytes": 100,
                "checksum": "abc123",
            },
        }

        with patch("core.langgraph.nodes.commit_node.ContentManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm_class.return_value = mock_cm
            mock_cm.read_json.return_value = {
                "characters": [],
                "world_items": [],
            }

            result = await commit_to_graph(mock_state)  # type: ignore[arg-type]

            # Should return a valid state dict with expected fields
            assert isinstance(result, dict)
            assert result["current_node"] == "commit_to_graph"
            assert result["has_fatal_error"] is False
