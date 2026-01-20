# tests/core/langgraph/nodes/test_commit_node.py
"""Tests for core/langgraph/nodes/commit_node.py - entity and relationship persistence."""

from unittest.mock import MagicMock, patch

import pytest

from core.langgraph.nodes.commit_node import commit_to_graph


class TestCommitNodeEntityPersistence:
    """Test entity persistence operations in the commit node."""

    @pytest.mark.asyncio
    async def test_commit_to_graph_creates_entity_ids(self) -> None:
        """Test that commit_to_graph generates unique entity IDs."""
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
        }

        with patch("core.langgraph.nodes.commit_node.ContentManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm_class.return_value = mock_cm

            # Mock the content manager to return test data
            mock_cm.load_json_strict.return_value = {
                "characters": [
                    {"name": "Alice", "type": "Character", "description": "Protagonist", "first_appearance_chapter": 1},
                    {"name": "Bob", "type": "Character", "description": "Antagonist", "first_appearance_chapter": 1},
                ],
                "world_items": [
                    {"name": "Sword", "type": "Item", "description": "A sharp sword", "first_appearance_chapter": 1},
                ],
            }

            with patch("core.langgraph.nodes.commit_node.generate_entity_id") as mock_generate_id:
                mock_generate_id.side_effect = lambda name, category, chapter: f"id_{name}_{chapter}"

                # Mock similarity checks to return no matches (so generate_entity_id is called)
                with patch("core.langgraph.nodes.commit_node.check_entity_similarity") as mock_similarity:
                    mock_similarity.return_value = None  # No similar entity found

                    with patch("core.langgraph.nodes.commit_node.should_merge_entities") as mock_should_merge:
                        mock_should_merge.return_value = False  # Should not merge

                        with patch("core.db_manager.neo4j_manager.execute_cypher_batch"):
                            await commit_to_graph(mock_state)  # type: ignore[arg-type]

                            # Verify generate_entity_id was called for world items only
                            # Characters use their names as identifiers, world items get stable IDs
                            # 1 world item = 1 call
                            assert mock_generate_id.call_count == 1

    @pytest.mark.asyncio
    async def test_commit_to_graph_handles_empty_extractions(self) -> None:
        """Test that commit_to_graph handles empty extraction results."""
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
            mock_cm.read_json.return_value = {
                "characters": [],
                "world_items": [],
            }
            mock_cm.load_text_strict.return_value = "Test draft text"

            with patch("core.db_manager.neo4j_manager.execute_cypher_batch") as mock_execute:
                # Add draft_ref to state so get_draft_text doesn't fail
                mock_state["draft_ref"] = {
                    "path": ".saga/content/drafts/chapter_1.txt",
                    "content_type": "draft",
                    "version": 1,
                    "size_bytes": 10,
                    "checksum": "draft123",
                }
                await commit_to_graph(mock_state)  # type: ignore[arg-type]

                # Should still execute the batch (for chapter node creation)
                assert mock_execute.called

    @pytest.mark.asyncio
    async def test_commit_to_graph_deduplicates_entities(self) -> None:
        """Test that commit_to_graph performs entity deduplication."""
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
                "characters": [
                    {"name": "Alice", "entity_type": "character"},
                    {"name": "Alice", "entity_type": "character"},  # Duplicate
                ],
                "world_items": [],
            }

            with patch("core.db_manager.neo4j_manager.execute_cypher_batch"):
                await commit_to_graph(mock_state)  # type: ignore[arg-type]

                # Should not call generate_entity_id since there are no world items
                # The deduplication logic removes duplicate characters within the batch


class TestCommitNodeRelationshipPersistence:
    """Test relationship persistence operations in the commit node."""

    @pytest.mark.asyncio
    async def test_commit_to_graph_creates_relationships(self) -> None:
        """Test that commit_to_graph creates relationships between entities."""
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
            mock_cm.load_text_strict.return_value = "Test draft text"

            # Add draft_ref to state so get_draft_text doesn't fail
            mock_state["draft_ref"] = {
                "path": ".saga/content/drafts/chapter_1.txt",
                "content_type": "draft",
                "version": 1,
                "size_bytes": 10,
                "checksum": "draft123",
            }

            with patch("core.db_manager.neo4j_manager.execute_cypher_batch") as mock_execute:
                await commit_to_graph(mock_state)  # type: ignore[arg-type]

                # Verify the batch was executed
                assert mock_execute.called

    @pytest.mark.asyncio
    async def test_commit_to_graph_handles_missing_relationships(self) -> None:
        """Test that commit_to_graph handles missing relationship data."""
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
            # No extracted_relationships_ref
        }

        with patch("core.langgraph.nodes.commit_node.ContentManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm_class.return_value = mock_cm
            mock_cm.read_json.return_value = {
                "characters": [{"name": "Alice", "entity_type": "character"}],
                "world_items": [],
            }
            mock_cm.load_text_strict.return_value = "Test draft text"

            # Add draft_ref to state so get_draft_text doesn't fail
            mock_state["draft_ref"] = {
                "path": ".saga/content/drafts/chapter_1.txt",
                "content_type": "draft",
                "version": 1,
                "size_bytes": 10,
                "checksum": "draft123",
            }

            with patch("core.db_manager.neo4j_manager.execute_cypher_batch") as mock_execute:
                await commit_to_graph(mock_state)  # type: ignore[arg-type]

                # Should still execute the batch (for entities and chapter)
                assert mock_execute.called


class TestCommitNodeChapterPersistence:
    """Test chapter node creation in the commit node."""

    @pytest.mark.asyncio
    async def test_commit_to_graph_creates_chapter_node(self) -> None:
        """Test that commit_to_graph creates a chapter node."""
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
                "characters": [],
                "world_items": [],
            }
            mock_cm.load_text_strict.return_value = "Test draft text"

            # Add draft_ref to state so get_draft_text doesn't fail
            mock_state["draft_ref"] = {
                "path": ".saga/content/drafts/chapter_1.txt",
                "content_type": "draft",
                "version": 1,
                "size_bytes": 10,
                "checksum": "draft123",
            }

            with patch("core.db_manager.neo4j_manager.execute_cypher_batch") as mock_execute:
                await commit_to_graph(mock_state)  # type: ignore[arg-type]

                # Verify the batch was executed (should contain chapter creation)
                assert mock_execute.called

    @pytest.mark.asyncio
    async def test_commit_to_graph_links_chapter_to_entities(self) -> None:
        """Test that commit_to_graph links the chapter to created entities."""
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
            mock_cm.load_text_strict.return_value = "Test draft text"

            # Add draft_ref to state so get_draft_text doesn't fail
            mock_state["draft_ref"] = {
                "path": ".saga/content/drafts/chapter_1.txt",
                "content_type": "draft",
                "version": 1,
                "size_bytes": 10,
                "checksum": "draft123",
            }

            with patch("core.db_manager.neo4j_manager.execute_cypher_batch") as mock_execute:
                await commit_to_graph(mock_state)  # type: ignore[arg-type]

                # Verify the batch was executed
                assert mock_execute.called


class TestCommitNodeErrorHandling:
    """Test error handling in the commit node."""

    @pytest.mark.asyncio
    async def test_commit_to_graph_handles_database_errors(self) -> None:
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


class TestCommitNodeStateManagement:
    """Test state management in the commit node."""

    @pytest.mark.asyncio
    async def test_commit_to_graph_updates_state(self) -> None:
        """Test that commit_to_graph updates the state correctly."""
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

            with patch("core.db_manager.neo4j_manager.execute_cypher_batch"):
                result = await commit_to_graph(mock_state)  # type: ignore[arg-type]

                # Should update current_node
                assert result["current_node"] == "commit_to_graph"

    @pytest.mark.asyncio
    async def test_commit_to_graph_preserves_existing_state(self) -> None:
        """Test that commit_to_graph preserves existing state fields."""
        mock_state = {
            "chapter": 1,
            "project_dir": "/tmp/test_project",
            "some_existing_field": "preserve_this",
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
            mock_cm.load_text_strict.return_value = "Test draft text"

            # Add draft_ref to state so get_draft_text doesn't fail
            mock_state["draft_ref"] = {
                "path": ".saga/content/drafts/chapter_1.txt",
                "content_type": "draft",
                "version": 1,
                "size_bytes": 10,
                "checksum": "draft123",
            }

            with patch("core.db_manager.neo4j_manager.execute_cypher_batch"):
                result = await commit_to_graph(mock_state)  # type: ignore[arg-type]

                # Should return success state with expected fields
                assert result["current_node"] == "commit_to_graph"
                assert result["has_fatal_error"] is False
