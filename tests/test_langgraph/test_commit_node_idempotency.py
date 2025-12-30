# tests/test_langgraph/test_commit_node_idempotency.py
from unittest.mock import patch

import pytest

from core.langgraph.nodes.commit_node import _build_relationship_statements, commit_to_graph
from core.langgraph.state import ExtractedEntity, ExtractedRelationship


@pytest.mark.asyncio
class TestCommitNodeIdempotency:
    """Tests to ensure chapter-idempotent KG writes."""

    async def test_build_relationship_statements_includes_delete(self):
        """Verify that _build_relationship_statements always prepends a DELETE statement."""
        relationships = [ExtractedRelationship(source_name="Alice", target_name="Bob", relationship_type="FRIEND_OF", description="Friends", chapter=1, confidence=1.0)]
        char_entities = [
            ExtractedEntity(name="Alice", type="Character", description="Alice", first_appearance_chapter=1, attributes={}),
            ExtractedEntity(name="Bob", type="Character", description="Bob", first_appearance_chapter=1, attributes={}),
        ]

        with patch("core.langgraph.nodes.commit_node._get_cypher_labels", return_value=":Character"):
            statements = await _build_relationship_statements(
                relationships=relationships, char_entities=char_entities, world_entities=[], char_mappings={"Alice": "Alice", "Bob": "Bob"}, world_mappings={}, chapter=1, is_from_flawed_draft=False
            )

        # Should have at least the DELETE and one MERGE
        assert len(statements) >= 2

        # First statement should be the DELETE
        delete_query, params = statements[0]
        assert "DELETE r" in delete_query
        assert "chapter_added" in delete_query
        assert params["chapter"] == 1

        # Subsequent statements should be MERGEs
        merge_query, merge_params = statements[1]
        assert "apoc.merge.relationship" in merge_query

    async def test_build_relationship_statements_empty_list_still_deletes(self):
        """Verify that even with no relationships, a DELETE statement is produced for the chapter."""
        statements = await _build_relationship_statements(relationships=[], char_entities=[], world_entities=[], char_mappings={}, world_mappings={}, chapter=5, is_from_flawed_draft=False)

        assert len(statements) == 1
        delete_query, params = statements[0]
        assert "DELETE r" in delete_query
        assert params["chapter"] == 5

    async def test_commit_to_graph_idempotency_call_sequence(self):
        """Verify that commit_to_graph calls relationship builder and sends batch with DELETE."""
        state = {
            "project_dir": "test_project",
            "current_chapter": 2,
            "extracted_entities": {"characters": [], "world_items": []},
            "extracted_relationships": [{"source_name": "Alice", "target_name": "Bob", "relationship_type": "KNOWS", "description": "...", "chapter": 2, "confidence": 1.0}],
        }

        # Mock dependencies
        with (
            patch("core.langgraph.nodes.commit_node.ContentManager"),
            patch("core.langgraph.nodes.commit_node.get_extracted_entities", return_value={"characters": [], "world_items": []}),
            patch("core.langgraph.nodes.commit_node.get_extracted_relationships") as mock_get_rels,
            patch("core.langgraph.nodes.commit_node.get_draft_text", return_value="Draft"),
            patch("core.langgraph.nodes.commit_node.chapter_queries.build_chapter_upsert_statement", return_value=("QUERY", {})),
            patch("data_access.cache_coordinator.clear_kg_read_caches"),
            patch("data_access.cache_coordinator.clear_character_read_caches"),
            patch("data_access.cache_coordinator.clear_world_read_caches"),
            patch("core.langgraph.nodes.commit_node._run_phase2_deduplication", return_value={}),
            patch("core.db_manager.neo4j_manager.execute_cypher_batch") as mock_batch,
        ):
            mock_get_rels.return_value = state["extracted_relationships"]
            mock_batch.return_value = None

            # First commit
            await commit_to_graph(state)

            assert mock_batch.called
            first_call_statements = mock_batch.call_args[0][0]

            # Find the delete statement in the first batch
            delete_statements = [s for s in first_call_statements if "DELETE r" in s[0]]
            assert len(delete_statements) == 1
            assert delete_statements[0][1]["chapter"] == 2

            # Reset mock and call again with different relationships
            mock_batch.reset_mock()
            state["extracted_relationships"] = []
            mock_get_rels.return_value = []

            await commit_to_graph(state)

            assert mock_batch.called
            second_call_statements = mock_batch.call_args[0][0]

            # Even with no relationships, the second call must include the DELETE for the chapter
            delete_statements_2 = [s for s in second_call_statements if "DELETE r" in s[0]]
            assert len(delete_statements_2) == 1
            assert delete_statements_2[0][1]["chapter"] == 2
