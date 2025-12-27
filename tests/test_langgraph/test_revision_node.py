# tests/test_langgraph/test_revision_node.py
"""
Tests for LangGraph revision node behavior in the scene-first pipeline.

This suite validates that `revise_chapter()` produces externalized revision guidance and
clears stale chapter artifacts so that the workflow regenerates scenes.
"""

from unittest.mock import AsyncMock, patch

import pytest

from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.revision_node import revise_chapter
from core.langgraph.state import Contradiction, create_initial_state


@pytest.fixture
def sample_revision_state(tmp_path: pytest.TempPathFactory) -> dict:
    project_dir = str(tmp_path / "test-project")

    state = create_initial_state(
        project_id="test-project",
        title="Test Novel",
        genre="Fantasy",
        theme="Adventure",
        setting="Medieval world",
        target_word_count=80000,
        total_chapters=20,
        project_dir=project_dir,
        protagonist_name="Hero",
        generation_model="test-model",
        medium_model="test-model",
        revision_model="test-revision-model",
    )

    content_manager = ContentManager(project_dir)

    chapter_plan = [
        {
            "title": "Into the Forest",
            "pov_character": "Hero",
            "setting": "Dark forest",
            "characters_involved": ["Hero"],
            "plot_point": "The hero begins their journey",
            "conflict": "Unease and ominous signs",
            "outcome": "Hero continues forward",
        },
        {
            "title": "The Dragon Appears",
            "pov_character": "Hero",
            "setting": "Forest clearing",
            "characters_involved": ["Hero", "Dragon"],
            "plot_point": "The dragon threatens the hero",
            "conflict": "Dragon confrontation",
            "outcome": "Hero survives and retreats",
        },
    ]
    chapter_plan_ref = content_manager.save_json(chapter_plan, "chapter_plan", "chapter_1", 1)
    state["chapter_plan_ref"] = chapter_plan_ref

    chapter_outlines = {
        1: {
            "plot_point": "The hero begins their journey",
            "chapter_summary": "Introduction to the protagonist",
        }
    }
    chapter_outlines_ref = content_manager.save_json(chapter_outlines, "chapter_outlines", "all", 1)
    state["chapter_outlines_ref"] = chapter_outlines_ref

    state["contradictions"] = [
        Contradiction(
            type="character_trait",
            description="Hero acts cowardly, contradicting established brave trait",
            conflicting_chapters=[1],
            severity="major",
            suggested_fix="Revise to show Hero's courage",
        ),
        Contradiction(
            type="plot_consistency",
            description="Dragon appears without foreshadowing",
            conflicting_chapters=[1],
            severity="minor",
            suggested_fix="Add earlier hints of dragon presence",
        ),
    ]

    state["iteration_count"] = 0
    state["max_iterations"] = 3
    state["needs_revision"] = True

    state["scene_drafts_ref"] = {"path": "mock_scene_drafts.json"}
    state["scene_embeddings_ref"] = {"path": "mock_scene_embeddings.json"}
    state["draft_ref"] = {"path": "mock_draft.txt"}
    state["embedding_ref"] = {"path": "mock_embedding.json"}
    state["extracted_entities_ref"] = {"path": "mock_entities.json"}
    state["extracted_relationships_ref"] = {"path": "mock_relationships.json"}
    state["generated_embedding"] = [0.1, 0.2]

    return state


@pytest.fixture
def mock_llm_guidance() -> AsyncMock:
    with patch("core.langgraph.nodes.revision_node.llm_service") as mock_llm:
        mock_llm.async_call_llm = AsyncMock(return_value=("- Fix Scene 1: add foreshadowing.\n- Fix Scene 2: show courage.", {}))
        mock_llm.count_tokens = lambda text, model: 800
        yield mock_llm


@pytest.mark.asyncio
async def test_revise_chapter_sets_revision_guidance_and_clears_artifacts(
    sample_revision_state: dict,
    mock_llm_guidance: AsyncMock,
) -> None:
    result = await revise_chapter(sample_revision_state)

    assert result["has_fatal_error"] is False
    assert result["last_error"] is None
    assert result["current_node"] == "revise"
    assert result["iteration_count"] == 1
    assert result["needs_revision"] is False
    assert result["contradictions"] == []

    assert result["revision_guidance_ref"] is not None
    assert isinstance(result["revision_guidance_ref"], dict)
    assert isinstance(result["revision_guidance_ref"].get("path"), str)

    assert result["scene_drafts_ref"] is None
    assert result["scene_embeddings_ref"] is None
    assert result["draft_ref"] is None
    assert result["embedding_ref"] is None
    assert result["extracted_entities_ref"] is None
    assert result["extracted_relationships_ref"] is None
    assert result["generated_embedding"] is None
    assert result["current_scene_index"] == 0

    content_manager = ContentManager(sample_revision_state["project_dir"])
    guidance_text = content_manager.load_text(result["revision_guidance_ref"])
    assert guidance_text == "- Fix Scene 1: add foreshadowing.\n- Fix Scene 2: show courage."

    mock_llm_guidance.async_call_llm.assert_called_once()


@pytest.mark.asyncio
async def test_revise_chapter_guidance_generation_failure_sets_fatal_error(
    sample_revision_state: dict,
) -> None:
    with patch("core.langgraph.nodes.revision_node.llm_service") as mock_llm:
        mock_llm.count_tokens = lambda text, model: 800
        mock_llm.async_call_llm = AsyncMock(side_effect=RuntimeError("LLM exploded"))

        result = await revise_chapter(sample_revision_state)

    assert result["has_fatal_error"] is True
    assert result["error_node"] == "revise"
    assert result["current_node"] == "revise"
    assert result["last_error"] == "Revision guidance generation failed"
