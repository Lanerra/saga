# tests/core/langgraph/nodes/test_commit_node_simple.py
"""Simple integration tests for core/langgraph/nodes/commit_node.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.commit_node import commit_to_graph
from core.langgraph.state import NarrativeState
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager
from tests.fakes.service_context import configure_empty_entity_names, patch_service

pytestmark = pytest.mark.usefixtures("offline_commit_providers")


@pytest.fixture(autouse=True)
def known_entity_names(offline_commit_providers: FakeNeo4jManager) -> None:
    configure_empty_entity_names(offline_commit_providers)


class TestCommitNodeIntegration:
    """Integration tests for the commit node."""

    @pytest.mark.asyncio
    async def test_commit_to_graph_with_valid_state(self, tmp_path: Path, offline_commit_providers: FakeNeo4jManager) -> None:
        """Test that commit_to_graph handles valid state without errors."""
        manager = ContentManager(str(tmp_path))
        state: NarrativeState = {
            "current_chapter": 1, "project_dir": str(tmp_path),
            "draft_ref": manager.save_text("Alice entered the room.", "draft", "chapter_1", 1),
            "extracted_entities_ref": manager.save_json({
                "characters": [{"name": "Alice", "type": "Character", "description": "A scout", "first_appearance_chapter": 1}],
                "world_items": [],
            }, "extracted_entities", "chapter_1", 1),
        }

        result = await commit_to_graph(state)

        assert result == {"current_node": "commit_to_graph", "has_fatal_error": False, "last_error": None}
        assert len(offline_commit_providers.batch_statements) == 1
        assert offline_commit_providers.batch_statements[0][1][1]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_commit_to_graph_with_empty_extractions(self) -> None:
        """Test that commit_to_graph handles empty extractions."""
        mock_state = {
            "current_chapter": 1,
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
            mock_cm.load_json_strict.return_value = {
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
            "current_chapter": 1,
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
            mock_cm.load_json_strict.side_effect = [
                {
                    "characters": [{"name": "Alice", "type": "Character", "description": "A scout", "first_appearance_chapter": 1}],
                    "world_items": [],
                },
                [
                    {
                        "source_name": "Alice",
                        "target_name": "Bob",
                        "relationship_type": "KNOWS",
                        "description": "Acquaintance",
                        "chapter": 1,
                        "confidence": 0.9,
                    },
                ],
            ]

            result = await commit_to_graph(mock_state)  # type: ignore[arg-type]

            # Should return a valid state dict
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_commit_to_graph_handles_database_error(self, tmp_path: Path) -> None:
        """Test that commit_to_graph handles database errors gracefully."""
        manager = ContentManager(str(tmp_path))
        state: NarrativeState = {
            "current_chapter": 1, "project_dir": str(tmp_path),
            "draft_ref": manager.save_text("Alice entered the room.", "draft", "chapter_1", 1),
            "extracted_entities_ref": manager.save_json({
                "characters": [{"name": "Alice", "type": "Character", "description": "A scout", "first_appearance_chapter": 1}],
                "world_items": [],
            }, "extracted_entities", "chapter_1", 1),
        }

        with patch_service('database.execute_cypher_batch') as execute:
            execute.side_effect = Exception("Database error")
            result = await commit_to_graph(state)

        execute.assert_awaited_once()
        assert result == {"current_node": "commit_to_graph", "has_fatal_error": True, "error_node": "commit", "last_error": "Commit to graph failed: Database error"}

    @pytest.mark.asyncio
    async def test_commit_to_graph_handles_missing_content(self) -> None:
        """Test that commit_to_graph handles missing content references."""
        mock_state = {
            "current_chapter": 1,
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
            "current_chapter": 1,
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
            mock_cm.load_json_strict.return_value = {
                "characters": [],
                "world_items": [],
            }
            mock_cm.load_text_strict.return_value = "Synthetic draft"

            result = await commit_to_graph(mock_state)  # type: ignore[arg-type]

            # Should return a valid state dict with expected fields
            assert isinstance(result, dict)
            assert result["current_node"] == "commit_to_graph"
            assert result["has_fatal_error"] is False
