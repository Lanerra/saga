# tests/core/langgraph/subgraphs/test_generation_subgraph.py
from unittest.mock import AsyncMock, patch

import pytest

from core.langgraph.state import create_initial_state
from core.langgraph.subgraphs.generation import create_generation_subgraph


@pytest.mark.asyncio
async def test_generation_subgraph_flow():
    # Setup mocks
    with (
        patch("core.langgraph.nodes.scene_planning_node.llm_service") as mock_llm_plan,
        patch(
            "core.langgraph.nodes.scene_generation_node.llm_service"
        ) as mock_llm_draft,
        patch(
            "core.langgraph.nodes.context_retrieval_node.get_reliable_kg_facts_for_drafting_prompt",
            new_callable=AsyncMock,
        ) as mock_kg,
    ):
        # Mock planning response
        mock_llm_plan.async_call_llm = AsyncMock(
            return_value=(
                '[{"title": "Scene 1", "pov_character": "Hero", "setting": "Room", "characters": ["Hero"], "plot_point": "Start", "conflict": "None", "outcome": "Next"}, '
                '{"title": "Scene 2", "pov_character": "Hero", "setting": "Outside", "characters": ["Hero"], "plot_point": "End", "conflict": "None", "outcome": "Done"}]',
                {},
            )
        )

        # Mock drafting response
        mock_llm_draft.async_call_llm = AsyncMock(
            side_effect=[("Draft for Scene 1", {}), ("Draft for Scene 2", {})]
        )

        # Mock KG response
        mock_kg.return_value = "KG Context"

        # Create graph
        graph = create_generation_subgraph()

        # Create initial state
        state = create_initial_state(
            project_id="test",
            title="Test Novel",
            genre="Sci-Fi",
            theme="Testing",
            setting="Lab",
            target_word_count=1000,
            total_chapters=1,
            project_dir="/tmp",
            protagonist_name="Hero",
        )
        state["chapter_outlines"] = {
            1: {"scene_description": "Test Chapter", "key_beats": ["Beat 1", "Beat 2"]}
        }

        # Run graph
        result = await graph.ainvoke(state)

        # Verify results
        assert result["draft_text"] == "Draft for Scene 1\n\n# ***\n\nDraft for Scene 2"
        assert len(result["scene_drafts"]) == 2
        assert result["current_scene_index"] == 2

        # Verify calls
        assert mock_llm_plan.async_call_llm.called
        assert mock_llm_draft.async_call_llm.call_count == 2
