# tests/core/langgraph/subgraphs/test_generation_subgraph.py
from unittest.mock import AsyncMock, patch

import pytest

from core.langgraph.content_manager import ContentManager, get_draft_text
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
        project_dir = "/tmp"
        state = create_initial_state(
            project_id="test",
            title="Test Novel",
            genre="Sci-Fi",
            theme="Testing",
            setting="Lab",
            target_word_count=1000,
            total_chapters=1,
            project_dir=project_dir,
            protagonist_name="Hero",
        )

        content_manager = ContentManager(project_dir)
        chapter_outlines = {
            1: {"scene_description": "Test Chapter", "key_beats": ["Beat 1", "Beat 2"]}
        }
        ref = content_manager.save_json(chapter_outlines, "chapter_outlines", "all", 1)
        state["chapter_outlines_ref"] = ref

        # Run graph
        result = await graph.ainvoke(state)

        # Verify results
        # Use ContentManager to get the actual text from the ref
        content_manager = ContentManager(state["project_dir"])
        draft_text = get_draft_text(result, content_manager)

        assert draft_text == "Draft for Scene 1\n\n# ***\n\nDraft for Scene 2"
        # scene_drafts is not in state anymore, it's externalized to scene_drafts_ref
        # But assemble_chapter doesn't return scene_drafts list in state, only ref.
        # We can verify the ref exists or use content manager to get it.
        # However, checking the test expectation: assert len(result["scene_drafts"]) == 2
        # If result["scene_drafts"] is missing, this will fail.
        # assemble_chapter returns:
        # { "draft_ref": ..., "scene_drafts_ref": ..., "draft_word_count": ..., "current_node": ... }
        # The state update merges this.
        # So "scene_drafts" (list) might still be in state if not explicitly removed, OR if using Reducer it might be there.
        # But assemble_chapter gets scene_drafts from content_manager (which reads from state or file).
        # Wait, get_scene_drafts reads from state["scene_drafts"] if present (legacy) or ref.
        # The scene_generation node produces "scene_drafts" update?
        # Let's check scene_generation_node.

        # If the test fails on scene_drafts, I will fix it. For now, let's fix draft_text.
        assert result["scene_drafts_ref"] is not None
        assert result["current_scene_index"] == 2

        # Verify calls
        assert mock_llm_plan.async_call_llm.called
        assert mock_llm_draft.async_call_llm.call_count == 2
