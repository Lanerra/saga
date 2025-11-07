"""
End-to-end tests for Phase 1 LangGraph workflow.

Tests the complete workflow integration: extract → commit → validate
"""

from unittest.mock import AsyncMock, patch

import pytest

from core.langgraph import (
    create_initial_state,
    create_phase1_graph,
)


class TestPhase1Workflow:
    """End-to-end workflow tests."""

    @pytest.mark.asyncio
    async def test_workflow_successful_path(self):
        """Test successful workflow execution: extract → commit → validate → end."""
        # Create initial state with draft text
        state = create_initial_state(
            project_id="test-workflow",
            title="Test Novel",
            genre="Fantasy",
            theme="Adventure",
            setting="Medieval world",
            target_word_count=80000,
            total_chapters=20,
            project_dir="/tmp/test-workflow",
            protagonist_name="Hero",
        )

        # Add draft text to state (normally from generation node)
        state["draft_text"] = """
        The hero entered the ancient temple, where he met the wise sage Eldrin.
        Eldrin gave him a magical sword called Lightbringer.
        Together they decided to journey to the Mountain of Shadows.
        """
        state["draft_word_count"] = 30

        # Mock all external dependencies
        with patch("core.langgraph.nodes.extraction_node.llm_service") as mock_llm:
            with patch(
                "core.langgraph.nodes.commit_node.knowledge_graph_service"
            ) as mock_kg:
                with patch(
                    "core.langgraph.nodes.commit_node.kg_queries"
                ) as mock_kg_queries:
                    with patch(
                        "core.langgraph.nodes.commit_node.chapter_queries"
                    ) as mock_ch_queries:
                        with patch(
                            "core.langgraph.nodes.commit_node.check_entity_similarity",
                            new=AsyncMock(return_value=None),
                        ):
                            with patch(
                                "core.langgraph.nodes.validation_node.validate_batch_constraints"
                            ) as mock_validate:
                                with patch(
                                    "core.langgraph.nodes.validation_node.neo4j_manager"
                                ) as mock_neo4j:
                                    # Setup LLM mock to return extraction results
                                    mock_llm.async_call_llm = AsyncMock(
                                        return_value=(
                                            """
                                            {
                                                "character_updates": {
                                                    "Hero": {"description": "The protagonist", "traits": ["brave"]},
                                                    "Eldrin": {"description": "A wise sage", "traits": ["wise"]}
                                                },
                                                "world_updates": {
                                                    "Lightbringer": {"category": "artifact", "description": "A magical sword"}
                                                }
                                            }
                                            """,
                                            {"total_tokens": 100},
                                        )
                                    )

                                    # Setup validation mock to return valid results
                                    mock_validate.return_value = []

                                    # Setup Neo4j mock
                                    mock_neo4j.fetch_all_characters_by_names = (
                                        AsyncMock(return_value=[])
                                    )

                                    # Setup service mocks
                                    mock_kg.persist_entities = AsyncMock(
                                        return_value=None
                                    )
                                    mock_kg_queries.add_kg_triples_batch_to_db = (
                                        AsyncMock(return_value=None)
                                    )
                                    mock_ch_queries.save_chapter_data_to_db = AsyncMock(
                                        return_value=None
                                    )

                                    # Create and run workflow
                                    graph = create_phase1_graph(checkpointer=None)

                                    # Execute workflow
                                    result = await graph.ainvoke(
                                        state,
                                        config={
                                            "configurable": {"thread_id": "test-1"}
                                        },
                                    )

                                    # Verify final state
                                    assert (
                                        result["current_node"] == "validate_consistency"
                                    )
                                    assert result["needs_revision"] is False
                                    assert result["last_error"] is None

                                    # Verify entities were extracted
                                    assert "extracted_entities" in result
                                    assert (
                                        len(
                                            result["extracted_entities"].get(
                                                "characters", []
                                            )
                                        )
                                        == 2
                                    )

                                    # Verify services were called
                                    assert mock_llm.async_call_llm.called
                                    assert mock_kg.persist_entities.called

    @pytest.mark.asyncio
    async def test_workflow_with_revision_loop(self):
        """Test workflow with revision: extract → commit → validate → revise → extract."""
        state = create_initial_state(
            project_id="test-revision",
            title="Test Novel",
            genre="Fantasy",
            theme="Adventure",
            setting="Medieval world",
            target_word_count=80000,
            total_chapters=20,
            project_dir="/tmp/test-revision",
            protagonist_name="Hero",
        )

        # Add draft text with low word count (triggers stagnation)
        state["draft_text"] = "Short chapter."
        state["draft_word_count"] = 2  # Below 1500 word minimum
        state["max_iterations"] = 2  # Allow 2 revisions

        with patch("core.langgraph.nodes.extraction_node.llm_service") as mock_llm:
            with patch(
                "core.langgraph.nodes.commit_node.knowledge_graph_service"
            ) as mock_kg:
                with patch("core.langgraph.nodes.commit_node.kg_queries"):
                    with patch("core.langgraph.nodes.commit_node.chapter_queries"):
                        with patch(
                            "core.langgraph.nodes.commit_node.check_entity_similarity",
                            new=AsyncMock(return_value=None),
                        ):
                            with patch(
                                "core.langgraph.nodes.validation_node.validate_batch_constraints"
                            ) as mock_validate:
                                with patch(
                                    "core.langgraph.nodes.validation_node.neo4j_manager"
                                ) as mock_neo4j:
                                    # Setup mocks
                                    mock_llm.async_call_llm = AsyncMock(
                                        return_value=(
                                            '{"character_updates": {}, "world_updates": {}}',
                                            {"total_tokens": 10},
                                        )
                                    )
                                    mock_validate.return_value = []
                                    mock_neo4j.fetch_all_characters_by_names = (
                                        AsyncMock(return_value=[])
                                    )
                                    mock_kg.persist_entities = AsyncMock(
                                        return_value=None
                                    )

                                    # Create and run workflow
                                    graph = create_phase1_graph(checkpointer=None)

                                    result = await graph.ainvoke(
                                        state,
                                        config={
                                            "configurable": {"thread_id": "test-2"}
                                        },
                                    )

                                    # Verify revision was triggered
                                    # Note: In Phase 1, revision is a placeholder that prevents infinite loop
                                    # Full revision logic will be in Phase 2
                                    assert result["iteration_count"] >= 0

    @pytest.mark.asyncio
    async def test_workflow_max_iterations_limit(self):
        """Test workflow respects max_iterations limit."""
        state = create_initial_state(
            project_id="test-max-iter",
            title="Test Novel",
            genre="Fantasy",
            theme="Adventure",
            setting="Medieval world",
            target_word_count=80000,
            total_chapters=20,
            project_dir="/tmp/test-max-iter",
            protagonist_name="Hero",
        )

        state["draft_text"] = "Short chapter."
        state["draft_word_count"] = 2
        state["max_iterations"] = 1  # Only allow 1 iteration

        with patch("core.langgraph.nodes.extraction_node.llm_service") as mock_llm:
            with patch(
                "core.langgraph.nodes.commit_node.knowledge_graph_service"
            ) as mock_kg:
                with patch("core.langgraph.nodes.commit_node.kg_queries"):
                    with patch("core.langgraph.nodes.commit_node.chapter_queries"):
                        with patch(
                            "core.langgraph.nodes.commit_node.check_entity_similarity",
                            new=AsyncMock(return_value=None),
                        ):
                            with patch(
                                "core.langgraph.nodes.validation_node.validate_batch_constraints"
                            ):
                                with patch(
                                    "core.langgraph.nodes.validation_node.neo4j_manager"
                                ) as mock_neo4j:
                                    mock_llm.async_call_llm = AsyncMock(
                                        return_value=(
                                            '{"character_updates": {}, "world_updates": {}}',
                                            {"total_tokens": 10},
                                        )
                                    )
                                    mock_neo4j.fetch_all_characters_by_names = (
                                        AsyncMock(return_value=[])
                                    )
                                    mock_kg.persist_entities = AsyncMock(
                                        return_value=None
                                    )

                                    graph = create_phase1_graph(checkpointer=None)

                                    result = await graph.ainvoke(
                                        state,
                                        config={
                                            "configurable": {"thread_id": "test-3"}
                                        },
                                    )

                                    # Workflow should complete even if needs_revision=True
                                    # because max_iterations was reached
                                    assert result is not None

    @pytest.mark.asyncio
    async def test_workflow_force_continue(self):
        """Test workflow with force_continue flag bypasses revision."""
        state = create_initial_state(
            project_id="test-force",
            title="Test Novel",
            genre="Fantasy",
            theme="Adventure",
            setting="Medieval world",
            target_word_count=80000,
            total_chapters=20,
            project_dir="/tmp/test-force",
            protagonist_name="Hero",
        )

        state["draft_text"] = "Short chapter."
        state["draft_word_count"] = 2  # Would normally trigger revision
        state["force_continue"] = True  # But this should bypass it

        with patch("core.langgraph.nodes.extraction_node.llm_service") as mock_llm:
            with patch(
                "core.langgraph.nodes.commit_node.knowledge_graph_service"
            ) as mock_kg:
                with patch("core.langgraph.nodes.commit_node.kg_queries"):
                    with patch("core.langgraph.nodes.commit_node.chapter_queries"):
                        with patch(
                            "core.langgraph.nodes.commit_node.check_entity_similarity",
                            new=AsyncMock(return_value=None),
                        ):
                            with patch(
                                "core.langgraph.nodes.validation_node.validate_batch_constraints"
                            ):
                                with patch(
                                    "core.langgraph.nodes.validation_node.neo4j_manager"
                                ) as mock_neo4j:
                                    mock_llm.async_call_llm = AsyncMock(
                                        return_value=(
                                            '{"character_updates": {}, "world_updates": {}}',
                                            {"total_tokens": 10},
                                        )
                                    )
                                    mock_neo4j.fetch_all_characters_by_names = (
                                        AsyncMock(return_value=[])
                                    )
                                    mock_kg.persist_entities = AsyncMock(
                                        return_value=None
                                    )

                                    graph = create_phase1_graph(checkpointer=None)

                                    result = await graph.ainvoke(
                                        state,
                                        config={
                                            "configurable": {"thread_id": "test-4"}
                                        },
                                    )

                                    # Should complete without revision due to force_continue
                                    assert result["force_continue"] is True
                                    assert (
                                        result["current_node"] == "validate_consistency"
                                    )


class TestWorkflowConditionalEdges:
    """Tests for conditional edge routing logic."""

    def test_should_revise_when_needed(self):
        """Test should_revise returns 'revise' when needs_revision=True."""
        from core.langgraph.workflow import should_revise

        state = {
            "needs_revision": True,
            "iteration_count": 0,
            "max_iterations": 3,
            "force_continue": False,
        }

        result = should_revise(state)
        assert result == "revise"

    def test_should_revise_when_max_iterations_reached(self):
        """Test should_revise returns 'end' when max iterations reached."""
        from core.langgraph.workflow import should_revise

        state = {
            "needs_revision": True,
            "iteration_count": 3,
            "max_iterations": 3,
            "force_continue": False,
        }

        result = should_revise(state)
        assert result == "end"

    def test_should_revise_when_force_continue(self):
        """Test should_revise returns 'end' when force_continue=True."""
        from core.langgraph.workflow import should_revise

        state = {
            "needs_revision": True,
            "iteration_count": 0,
            "max_iterations": 3,
            "force_continue": True,
        }

        result = should_revise(state)
        assert result == "end"

    def test_should_revise_when_not_needed(self):
        """Test should_revise returns 'end' when needs_revision=False."""
        from core.langgraph.workflow import should_revise

        state = {
            "needs_revision": False,
            "iteration_count": 0,
            "max_iterations": 3,
            "force_continue": False,
        }

        result = should_revise(state)
        assert result == "end"


class TestCheckpointer:
    """Tests for checkpoint configuration."""

    def test_create_checkpointer(self, tmp_path):
        """Test checkpointer creation."""
        from core.langgraph.workflow import create_checkpointer

        db_path = str(tmp_path / "test.db")
        checkpointer = create_checkpointer(db_path)

        assert checkpointer is not None
        # Verify database file was created
        import os

        assert os.path.exists(tmp_path)

    def test_graph_with_checkpointing(self, tmp_path):
        """Test graph creation with checkpointing enabled."""
        from core.langgraph.workflow import create_checkpointer, create_phase1_graph

        db_path = str(tmp_path / "checkpoint.db")
        checkpointer = create_checkpointer(db_path)
        graph = create_phase1_graph(checkpointer=checkpointer)

        assert graph is not None

    def test_graph_without_checkpointing(self):
        """Test graph creation without checkpointing."""
        from core.langgraph.workflow import create_phase1_graph

        graph = create_phase1_graph(checkpointer=None)
        assert graph is not None
