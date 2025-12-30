# tests/core/langgraph/subgraphs/test_generation_subgraph.py
from unittest.mock import AsyncMock, patch

import pytest

from core.exceptions import MissingDraftReferenceError
from core.langgraph.content_manager import ContentManager, get_draft_text, get_scene_drafts
from core.langgraph.state import create_initial_state
from core.langgraph.subgraphs.generation import create_generation_subgraph


@pytest.mark.asyncio
async def test_generation_subgraph_flow():
    with (
        patch("core.langgraph.nodes.scene_planning_node.llm_service") as mock_llm_plan,
        patch("core.langgraph.nodes.scene_generation_node.llm_service") as mock_llm_draft,
        patch(
            "core.langgraph.nodes.context_retrieval_node.get_reliable_kg_facts_for_drafting_prompt",
            new_callable=AsyncMock,
        ) as mock_kg,
    ):
        mock_llm_plan.async_call_llm = AsyncMock(
            return_value=(
                '[{"title": "Scene 1", "pov_character": "Hero", "setting": "Room", "characters": ["Hero"], "plot_point": "Start", "conflict": "None", "outcome": "Next"}, '
                '{"title": "Scene 2", "pov_character": "Hero", "setting": "Outside", "characters": ["Hero"], "plot_point": "End", "conflict": "None", "outcome": "Done"}]',
                {},
            )
        )
        mock_llm_draft.async_call_llm = AsyncMock(side_effect=[("Draft for Scene 1", {}), ("Draft for Scene 2", {})])
        mock_kg.return_value = "KG Context"

        graph = create_generation_subgraph()

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
        chapter_outlines = {1: {"scene_description": "Test Chapter", "key_beats": ["Beat 1", "Beat 2"]}}
        ref = content_manager.save_json(chapter_outlines, "chapter_outlines", "all", 1)
        state["chapter_outlines_ref"] = ref

        result = await graph.ainvoke(state)

        content_manager = ContentManager(state["project_dir"])

        with pytest.raises(MissingDraftReferenceError) as exc:
            get_draft_text(result, content_manager)
        assert str(exc.value) == "Missing required state key: draft_ref"

        assert result["scene_drafts_ref"] is not None
        assert result["current_scene_index"] == 2

        scene_drafts = get_scene_drafts(result, content_manager)
        assert scene_drafts == ["Draft for Scene 1", "Draft for Scene 2"]

        assert mock_llm_plan.async_call_llm.called
        assert mock_llm_draft.async_call_llm.call_count == 2
