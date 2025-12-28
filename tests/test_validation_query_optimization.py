"""Test validation query optimization - ensuring combined query reduces Neo4j round-trips."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.langgraph.subgraphs.validation import (
    _check_relationship_evolution,
    _check_timeline,
    _check_world_rules,
    _fetch_validation_data,
)


@pytest.mark.asyncio
async def test_fetch_validation_data_combined_query():
    """Test that _fetch_validation_data makes a single query and returns structured data."""

    # Mock Neo4j response with combined data
    mock_results = [
        # Events
        {"data_type": "event", "description": "Event 1", "timestamp": "2024-01-01", "chapter": 1},
        {"data_type": "event", "description": "Event 2", "timestamp": "2024-01-02", "chapter": 2},
        # World rules
        {"data_type": "world_rule", "description": "Magic rule", "constraint": "magic requires words", "created_chapter": 1},
        {"data_type": "world_rule", "description": "Physics rule", "constraint": "gravity exists", "created_chapter": 1},
        # Relationships
        {"data_type": "relationship", "source_name": "Alice", "target_name": "Bob", "rel_type": "FRIENDS_WITH", "first_chapter": 1},
        {"data_type": "relationship", "source_name": "Alice", "target_name": "Charlie", "rel_type": "HATES", "first_chapter": 2},
    ]

    with patch("core.langgraph.subgraphs.validation.neo4j_manager") as mock_manager:
        mock_manager.execute_read_query = AsyncMock(return_value=mock_results)

        result = await _fetch_validation_data(current_chapter=3)

        # Verify single query was made
        assert mock_manager.execute_read_query.call_count == 1

        # Verify structured data
        assert len(result["events"]) == 2
        assert len(result["world_rules"]) == 2
        assert len(result["relationships"]) == 2

        # Verify event data
        assert result["events"][0]["description"] == "Event 1"
        assert result["events"][0]["timestamp"] == "2024-01-01"

        # Verify world rule data
        assert result["world_rules"][0]["description"] == "Magic rule"
        assert result["world_rules"][0]["constraint"] == "magic requires words"

        # Verify relationship data
        assert ("Alice", "Bob") in result["relationships"]
        assert result["relationships"][("Alice", "Bob")]["rel_type"] == "FRIENDS_WITH"


@pytest.mark.asyncio
async def test_check_timeline_uses_prefetched_data():
    """Test that _check_timeline uses pre-fetched data instead of querying Neo4j."""

    extracted_events = [type("Event", (), {"description": "New Event", "attributes": {"timestamp": "after 2024-01-01"}})()]

    existing_events = [
        {"description": "Old Event", "timestamp": "before 2024-01-01", "chapter": 1},
    ]

    with patch("core.langgraph.subgraphs.validation.neo4j_manager") as mock_manager:
        with patch("core.langgraph.subgraphs.validation._events_are_related") as mock_related:
            with patch("core.langgraph.subgraphs.validation._is_temporal_violation") as mock_violation:
                # Mock to indicate events are related and there's a temporal violation
                mock_related.return_value = True
                mock_violation.return_value = True

                # Should not call execute_read_query when existing_events is provided
                result = await _check_timeline(extracted_events=extracted_events, current_chapter=2, existing_events=existing_events)

                assert mock_manager.execute_read_query.call_count == 0
                assert len(result) > 0  # Should find a contradiction


@pytest.mark.asyncio
async def test_check_world_rules_uses_prefetched_data():
    """Test that _check_world_rules uses pre-fetched data instead of querying Neo4j."""

    draft_text = "The magic works without words."
    world_rules = ["Magic requires spoken words"]

    existing_world_rules = [
        {"description": "Magic requires spoken words", "constraint": "magic requires words", "created_chapter": 1},
    ]

    with patch("core.langgraph.subgraphs.validation.neo4j_manager") as mock_manager:
        with patch("core.langgraph.subgraphs.validation.llm_service") as mock_llm:
            # Mock LLM response
            mock_llm.async_call_llm = AsyncMock(return_value=('[{"description": "Magic used without words", "severity": "major"}]', None))

            # Should not call execute_read_query when existing_world_rules is provided
            result = await _check_world_rules(draft_text=draft_text, world_rules=world_rules, current_chapter=2, existing_world_rules=existing_world_rules)

            assert mock_manager.execute_read_query.call_count == 0
            assert len(result) > 0  # Should find a violation


@pytest.mark.asyncio
async def test_check_relationship_evolution_uses_prefetched_data():
    """Test that _check_relationship_evolution uses pre-fetched data instead of querying Neo4j."""

    extracted_relationships = [type("Relationship", (), {"source_name": "Alice", "target_name": "Bob", "relationship_type": "LOVES"})()]

    existing_relationships = {("Alice", "Bob"): {"rel_type": "HATES", "first_chapter": 1}}

    with patch("core.langgraph.subgraphs.validation.neo4j_manager") as mock_manager:
        # Should not call execute_read_query when existing_relationships is provided
        result = await _check_relationship_evolution(extracted_relationships=extracted_relationships, current_chapter=2, existing_relationships=existing_relationships)

        assert mock_manager.execute_read_query.call_count == 0
        assert len(result) > 0  # Should find a relationship shift


@pytest.mark.asyncio
async def test_validation_subgraph_reduces_queries():
    """Test that the validation subgraph now makes only one Neo4j query instead of three."""

    from core.langgraph.subgraphs.validation import detect_contradictions

    # Mock state with extracted data
    mock_state = {
        "current_chapter": 3,
        "extracted_events": [type("Event", (), {"description": "New Event", "attributes": {"timestamp": "after 2024-01-01"}})()],
        "extracted_world_rules": [],
        "extracted_relationships": [type("Relationship", (), {"source_name": "Alice", "target_name": "Bob", "relationship_type": "LOVES"})()],
        "current_world_rules": ["Magic requires words"],
    }

    # Mock Neo4j to return combined data
    mock_validation_data = {
        "events": [{"description": "Old Event", "timestamp": "before 2024-01-01", "chapter": 1}],
        "world_rules": [{"description": "Magic requires words", "constraint": "magic requires words", "created_chapter": 1}],
        "relationships": {("Alice", "Bob"): {"rel_type": "HATES", "first_chapter": 1}},
    }

    with patch("core.langgraph.subgraphs.validation.neo4j_manager") as mock_manager:
        with patch("core.langgraph.subgraphs.validation._fetch_validation_data") as mock_fetch:
            mock_fetch.return_value = mock_validation_data

            # Mock LLM for world rule checking
            with patch("core.langgraph.subgraphs.validation.llm_service") as mock_llm:
                mock_llm.async_call_llm = AsyncMock(
                    return_value=(
                        "[]",  # No violations
                        None,
                    )
                )

                # Mock content manager and require_project_dir
                with patch("core.langgraph.subgraphs.validation.ContentManager") as mock_cm_class:
                    mock_cm = MagicMock()
                    mock_cm_class.return_value = mock_cm

                    with patch("core.langgraph.subgraphs.validation.require_project_dir") as mock_require_dir:
                        mock_require_dir.return_value = "/tmp/test"

                        # Mock get_extracted_entities to return simple data
                        with patch("core.langgraph.subgraphs.validation.get_extracted_entities") as mock_get_entities:
                            with patch("core.langgraph.subgraphs.validation.get_extracted_events_for_validation") as mock_get_events:
                                with patch("core.langgraph.subgraphs.validation.get_draft_text") as mock_get_draft:
                                    with patch("core.langgraph.subgraphs.validation.get_extracted_relationships") as mock_get_rels:
                                        mock_get_entities.return_value = []
                                        mock_get_events.return_value = mock_state["extracted_events"]
                                        mock_get_draft.return_value = "Test draft text"
                                        mock_get_rels.return_value = mock_state["extracted_relationships"]

                                        result = await detect_contradictions(mock_state)

                                        # Should only call _fetch_validation_data once (which makes one query)
                                        assert mock_fetch.call_count == 1

                                        # Should not make additional queries for timeline, world rules, or relationships
                                        assert mock_manager.execute_read_query.call_count == 0

                                        # Should have detected contradictions
                                        assert "contradictions" in result
