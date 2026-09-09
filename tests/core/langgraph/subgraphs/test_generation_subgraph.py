# tests/core/langgraph/subgraphs/test_generation_subgraph.py
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from core.exceptions import MissingDraftReferenceError
from core.langgraph.content_manager import ContentManager, get_draft_text, get_scene_drafts
from core.langgraph.state import create_initial_state
from core.langgraph.subgraphs.generation import create_generation_subgraph
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager
from tests.fakes.service_context import patch_service


@pytest.mark.asyncio
@pytest.mark.usefixtures("offline_graph_reads")
async def test_generation_subgraph_flow(tmp_path: Path) -> None:
    with (
        patch_service('database', FakeNeo4jManager()),
        patch_service('language_model.async_get_embedding', new=AsyncMock(return_value=[0.25, 0.75])),
        patch_service('language_model.async_call_llm', new_callable=AsyncMock) as completions,
        patch(
            "core.langgraph.nodes.context_world_retrieval.get_reliable_kg_facts_for_drafting_prompt",
            new_callable=AsyncMock,
        ) as mock_kg,
    ):
        completions.side_effect = [
            (
                '[{"title": "Scene 1", "pov_character": "Hero", "setting": "Room", "characters": ["Hero"], "plot_point": "Start", "conflict": "None", "outcome": "Next", "beats": ["Setup"]}, '
                '{"title": "Scene 2", "pov_character": "Hero", "setting": "Outside", "characters": ["Hero"], "plot_point": "End", "conflict": "None", "outcome": "Done", "beats": ["Climax"]}]',
                {},
            ),
            ("Draft for Scene 1", {}),
            ("Draft for Scene 2", {}),
        ]
        mock_kg.return_value = "KG Context"

        graph = create_generation_subgraph()

        project_dir = str(tmp_path)
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

        assert completions.await_count == 3
