from unittest.mock import patch

import pytest

from core.langgraph.content_manager import ContentManager
from core.langgraph.state import create_initial_state


@pytest.mark.asyncio
async def test_scene_extraction_subgraph_runs_extraction_and_consolidation(tmp_path):
    from core.langgraph.subgraphs.scene_extraction import (
        create_scene_extraction_subgraph,
    )

    workflow = create_scene_extraction_subgraph()

    project_dir = str(tmp_path / "test_project")
    state = create_initial_state(
        project_id="test",
        title="Test Novel",
        genre="Fantasy",
        theme="Adventure",
        setting="World",
        target_word_count=50000,
        total_chapters=10,
        project_dir=project_dir,
        protagonist_name="Elara",
    )

    content_manager = ContentManager(project_dir)
    scenes = ["Elara enters the library.", "She finds the map."]
    state["scene_drafts_ref"] = content_manager.save_list_of_texts(
        scenes, "scenes", "chapter_1", 1
    )
    state["current_chapter"] = 1

    async def mock_llm(*args, **kwargs):
        return {"character_updates": {}, "world_updates": {"Location": {}, "Event": {}}, "kg_triples": []}, None

    with patch(
        "core.langgraph.nodes.scene_extraction.llm_service.async_call_llm_json_object",
        side_effect=mock_llm,
    ):
        result = await workflow.ainvoke(state)

    assert "extracted_entities" in result
    assert "extracted_relationships" in result
