# tests/core/langgraph/subgraphs/test_generation_subgraph_unittest.py
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from core.db_manager import neo4j_manager
from core.exceptions import MissingDraftReferenceError
from core.langgraph.content_manager import ContentManager, get_draft_text, get_scene_drafts
from core.langgraph.subgraphs.generation import create_generation_subgraph
from tests import test_generation_failure_contract as generation_contract
from tests.fakes.generation_boundary import GenerationDatabase
from tests.fakes.service_context import patch_service
from tests.test_r08g_catalog_fixtures import catalog_state

generation_boundaries = generation_contract.boundaries

class TestGenerationSubgraph:
    @pytest.mark.asyncio
    @pytest.mark.run_settings(TARGET_SCENES_MIN=2, TARGET_SCENES_MAX=2)
    async def test_generation_subgraph_flow(self, tmp_path: Path, generation_boundaries: tuple[object, GenerationDatabase]) -> None:
        with (
            patch_service('language_model.async_get_embedding', new=AsyncMock(return_value=[0.25, 0.75])),
            patch_service('language_model.async_call_llm', new_callable=AsyncMock) as completions,
            patch(
                "core.langgraph.nodes.context_world_retrieval.get_reliable_kg_facts_for_drafting_prompt",
                new_callable=AsyncMock,
            ) as mock_kg,
        ):
            completions.side_effect = [
                (
                    '[{"title": "Scene 1", "pov_character": "Hero", "setting": "Room", "characters": ["Hero"], "plot_point": "Start", "conflict": "None", "outcome": "Next", "beats": ["Open door"]}, '
                    '{"title": "Scene 2", "pov_character": "Hero", "setting": "Room", "characters": ["Hero"], "plot_point": "End", "conflict": "None", "outcome": "Done", "beats": ["Cross room"]}]',
                    {},
                ),
                ("Draft for Scene 1", {}),
                ("Draft for Scene 2", {}),
            ]

            mock_kg.return_value = "KG Context"

            graph = create_generation_subgraph()

            state = catalog_state(tmp_path, characters=("Hero",), locations=("Room",), events=("Open door", "Cross room"), existing={"theme": "Testing", "protagonist_name": "Hero"})
            database = generation_boundaries[1]
            database.select(state)
            neo4j_manager.bind_project(state["graph_project_id"])
            content_manager = ContentManager(str(tmp_path))
            assert database.profiles["Hero"]["name"] == "Hero"

            result = await graph.ainvoke(state)
            assert not result.get("last_error"), result

            with pytest.raises(MissingDraftReferenceError):
                get_draft_text(result, content_manager)

            assert result["scene_drafts_ref"] is not None
            assert result["current_scene_index"] == 2

            scene_drafts = get_scene_drafts(result, content_manager)
            assert scene_drafts == ["Draft for Scene 1", "Draft for Scene 2"]

            assert completions.await_count == 3
