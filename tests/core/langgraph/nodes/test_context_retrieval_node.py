"""Tests for core/langgraph/nodes/context_retrieval_node.py.

Focuses on pure functions that do not require external services, plus
a main-flow integration test with patched dependencies.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes import context_character_retrieval, context_retrieval_node, context_scene_retrieval, context_world_retrieval
from core.langgraph.nodes.context_retrieval_node import retrieve_context
from core.langgraph.nodes.context_scene_retrieval import _smart_truncate_scene
from core.langgraph.state import NarrativeState
from core.service_context import get_services
from processing.scene_plan_parser import extract_scene_characters as _extract_scene_characters


async def test_retrieval_helpers_build_and_externalize_real_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content_manager = ContentManager(str(tmp_path))
    scene = {"title": "Arrival", "characters": ["Alice", " Alice "], "location": "Castle"}
    state: NarrativeState = {
        "project_dir": str(tmp_path), "current_chapter": 2, "current_scene_index": 1, "narrative_model": "synthetic",
        "chapter_plan_ref": content_manager.save_json([{"title": "Opening"}, scene], "chapter_plan", "chapter_2", 1),
        "chapter_outlines_ref": content_manager.save_json({"2": {"act_number": 1}}, "chapter_outlines", "outline", 1),
        "scene_drafts_ref": content_manager.save_json(["Alice approached."], "scene_drafts", "chapter_2", 1),
        "summaries_ref": content_manager.save_json(["An earlier journey."], "summaries", "recent", 1),
    }
    profiles = AsyncMock(return_value="Alice: a scout")
    facts = AsyncMock(return_value="Alice seeks shelter.")
    monkeypatch.setattr(context_character_retrieval, "get_filtered_character_profiles_for_prompt_plain_text", profiles)
    monkeypatch.setattr(context_world_retrieval, "get_reliable_kg_facts_for_drafting_prompt", facts)
    query_responses = {
        "get_scene_events": [{"name": "Arrival", "description": "Gates open"}],
        "get_character_relationships_for_scene": [{"source": "Alice", "target": "Bob", "relationship_type": "FRIEND_OF", "description": "Companions"}],
        "get_character_items": [{"character_name": "Alice", "item_name": "Map", "item_description": "A route"}],
        "get_scene_items": [{"item_name": "Key", "item_description": "Iron"}],
        "get_act_events": {"major_plot_points": [{"name": "Quest", "description": "Find shelter", "sequence_order": 1}], "act_key_events": []},
    }
    for name, response in query_responses.items():
        monkeypatch.setattr(f"data_access.scene_queries.{name}", AsyncMock(return_value=response))
    monkeypatch.setattr("data_access.kg_queries.query_kg_from_db", AsyncMock(return_value=[{"predicate": "HAS", "object": "Gates"}]))
    semantic_search = AsyncMock(return_value=[{"chapter_number": 1, "summary": "Past journey", "context_type": "immediate_previous"}])
    monkeypatch.setattr("data_access.chapter_queries.find_semantic_context_native", semantic_search)
    embedding = AsyncMock(return_value=[0.25, 0.75])
    monkeypatch.setattr(get_services().language_model, "async_get_embedding", embedding)

    def count_words(text: str, model_name: str) -> int:
        return len(text.split())

    def bounded_words(*, text: str, model_name: str, max_tokens: int, truncation_marker: str) -> str:
        assert count_words(text, model_name) <= max_tokens
        return text

    for module in (context_retrieval_node, context_scene_retrieval):
        monkeypatch.setattr(module, "count_tokens", count_words)
    for module in (context_character_retrieval, context_scene_retrieval, context_world_retrieval):
        monkeypatch.setattr(module, "truncate_text_by_tokens", bounded_words)
    original_state = dict(state)

    update = await retrieve_context(state)

    assert set(update) == {"current_node", "hybrid_context_ref"}
    assert update["current_node"] == "retrieve_context"
    assert state == original_state
    assert update["hybrid_context_ref"] is not None
    assert content_manager.load_text_strict(update["hybrid_context_ref"]) == "\n\n".join([
        "**Scene Character Profiles:**\nAlice: a scout",
        "Alice seeks shelter.",
        "**Scene Events:**\n\n- **Arrival**: Gates open",
        "**Character Relationships:**\n\n- Alice friend of Bob: Companions",
        "**Character Possessions:**\n\n- Alice:\n  - Map: A route",
        "**Items Featured in Scene:**\n\n- Key: Iron",
        "**Act 1 Plot Structure:**\n\nMajor Plot Points:\n- Quest: Find shelter\n",
        "\n\n**Recent Chapter Summaries:**\n\nAn earlier journey.",
        "\n\n**Previous Scenes in This Chapter:**\n\n--- Opening ---\nAlice approached.\n",
        "**Current Location - Castle:**\n- Castle has: Gates",
        "**Relevant Past Context (Semantic Search):**\n\n--- Chapter 1 (Previous Chapter) ---\nPast journey",
    ])
    profiles.assert_awaited_once_with(character_names=["Alice"], up_to_chapter_inclusive=1)
    assert facts.await_args is not None
    assert facts.await_args.kwargs["chapter_plan"] == [{"characters_involved": ["Alice"], **scene}]
    embedding.assert_awaited_once_with("Arrival   ")
    semantic_search.assert_awaited_once_with(query_embedding=[0.25, 0.75], embedding_model=get_services().configuration.EMBEDDING_MODEL, current_chapter_number=2, limit=3)


class TestExtractSceneCharacters:
    """Tests for the _extract_scene_characters helper."""

    def test_basic_extraction(self) -> None:
        scene = {"characters": ["Alice", "Bob"]}
        result = _extract_scene_characters(scene)
        assert result == ["Alice", "Bob"]

    def test_deduplication_preserves_order(self) -> None:
        scene = {"characters": ["Alice", "Bob", "Alice", "Carol", "Bob"]}
        result = _extract_scene_characters(scene)
        assert result == ["Alice", "Bob", "Carol"]

    def test_strips_whitespace(self) -> None:
        scene = {"characters": ["  Alice  ", " Bob ", "  Carol"]}
        result = _extract_scene_characters(scene)
        assert result == ["Alice", "Bob", "Carol"]

    def test_empty_characters_list(self) -> None:
        scene: dict[str, list[str]] = {"characters": []}
        result = _extract_scene_characters(scene)
        assert result == []

    def test_missing_characters_key(self) -> None:
        scene = {"title": "Opening Scene"}
        result = _extract_scene_characters(scene)
        assert result == []

    def test_blank_strings_filtered_out(self) -> None:
        scene = {"characters": ["Alice", "", "  ", "Bob"]}
        result = _extract_scene_characters(scene)
        assert result == ["Alice", "Bob"]

    def test_single_character(self) -> None:
        scene = {"characters": ["Aria"]}
        result = _extract_scene_characters(scene)
        assert result == ["Aria"]

    def test_deduplication_after_strip(self) -> None:
        scene = {"characters": ["Alice", " Alice ", "Alice"]}
        result = _extract_scene_characters(scene)
        assert result == ["Alice"]


class TestSmartTruncateScene:
    """Tests for the _smart_truncate_scene helper."""

    def test_short_text_returned_unchanged(self) -> None:
        text = "A short scene."
        with patch(
            "core.langgraph.nodes.context_scene_retrieval.count_tokens",
            return_value=5,
        ):
            result = _smart_truncate_scene(text, "test-model", max_tokens=100)
        assert result == "A short scene."

    def test_long_text_contains_head_and_tail(self) -> None:
        words = [f"word{i}" for i in range(200)]
        text = " ".join(words)

        with patch(
            "core.langgraph.nodes.context_scene_retrieval.count_tokens",
            return_value=200,
        ):
            result = _smart_truncate_scene(text, "test-model", max_tokens=50)

        assert "[...]" in result
        assert result.startswith("word0")
        assert result.endswith(f"word{199}")

    def test_truncation_preserves_head_and_tail_sections(self) -> None:
        words = [f"w{i}" for i in range(1000)]
        text = " ".join(words)

        with patch(
            "core.langgraph.nodes.context_scene_retrieval.count_tokens",
            return_value=1000,
        ):
            result = _smart_truncate_scene(text, "test-model", max_tokens=100)

        parts = result.split("[...]")
        assert len(parts) == 2
        head_words = parts[0].strip().split()
        tail_words = parts[1].strip().split()
        assert head_words == words[:10]
        assert tail_words == words[-80:]
        assert result == f"{' '.join(words[:10])}\n[...]\n{' '.join(words[-80:])}"

    def test_exact_budget_returns_unchanged(self) -> None:
        text = "exactly on budget text"
        with patch(
            "core.langgraph.nodes.context_scene_retrieval.count_tokens",
            return_value=50,
        ):
            result = _smart_truncate_scene(text, "test-model", max_tokens=50)
        assert result == "exactly on budget text"

    def test_head_portion_is_smaller_than_tail(self) -> None:
        words = [f"w{i}" for i in range(500)]
        text = " ".join(words)

        with patch(
            "core.langgraph.nodes.context_scene_retrieval.count_tokens",
            return_value=500,
        ):
            result = _smart_truncate_scene(text, "test-model", max_tokens=100)

        parts = result.split("[...]")
        head_section = parts[0].strip()
        tail_section = parts[1].strip()
        head_word_count = len(head_section.split())
        tail_word_count = len(tail_section.split())
        assert tail_word_count > head_word_count


class TestRetrieveContext:
    """Integration-level tests for the retrieve_context main flow."""

    async def test_invalid_scene_index_returns_current_node(self, tmp_path: Path) -> None:
        state: NarrativeState = {
            "project_dir": str(tmp_path),
            "current_chapter": 1,
            "current_scene_index": 99,
        }

        with (
            patch(
                "core.langgraph.nodes.context_retrieval_node.get_chapter_plan",
                return_value=[],
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node.ContentManager",
            ),
        ):
            from core.langgraph.nodes.context_retrieval_node import retrieve_context

            result = await retrieve_context(state)

        assert result["current_node"] == "retrieve_context"
        assert "has_fatal_error" not in result

    async def test_successful_context_build(self, tmp_path: Path) -> None:
        state: NarrativeState = {
            "project_dir": str(tmp_path),
            "current_chapter": 2,
            "current_scene_index": 0,
            "narrative_model": "test-model",
        }

        fake_scene = {
            "title": "Opening",
            "characters": ["Alice"],
            "plot_point": "Alice arrives",
            "conflict": "",
            "setting": "Forest",
            "location": "Dark Forest",
        }

        fake_content_manager = MagicMock()
        fake_content_manager.get_latest_version.return_value = 0
        fake_content_manager.save_text.return_value = {
            "path": ".saga/content/hybrid_context/chapter_2_scene_0_v1.txt",
            "content_type": "hybrid_context",
            "version": 1,
            "size_bytes": 100,
            "checksum": "abc123",
        }

        with (
            patch(
                "core.langgraph.nodes.context_retrieval_node.ContentManager",
                return_value=fake_content_manager,
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node.get_chapter_plan",
                return_value=[fake_scene],
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node.get_chapter_outlines",
                return_value={},
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node.get_previous_summaries",
                return_value=[],
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node.get_scene_drafts",
                return_value=[],
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node.get_scene_character_context",
                new_callable=AsyncMock,
                return_value="**Scene Character Profiles:**\nAlice: A brave warrior",
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node.get_scene_kg_facts",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node.get_scene_events_context",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node.get_character_relationships_context",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node.get_character_items_context",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node.get_scene_items_context",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node.get_location_context",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node.get_semantic_context",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node.count_tokens",
                return_value=50,
            ),
        ):
            from core.langgraph.nodes.context_retrieval_node import retrieve_context

            result = await retrieve_context(state)

        assert result["current_node"] == "retrieve_context"
        assert result["hybrid_context_ref"] is not None
        assert result["hybrid_context_ref"]["content_type"] == "hybrid_context"

    async def test_character_context_error_sets_fatal(self, tmp_path: Path) -> None:
        state: NarrativeState = {
            "project_dir": str(tmp_path),
            "current_chapter": 1,
            "current_scene_index": 0,
            "narrative_model": "test-model",
        }

        fake_scene = {
            "title": "Opening",
            "characters": ["Alice"],
            "plot_point": "",
            "conflict": "",
            "setting": "",
        }

        with (
            patch(
                "core.langgraph.nodes.context_retrieval_node.ContentManager",
                return_value=MagicMock(),
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node.get_chapter_plan",
                return_value=[fake_scene],
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node.get_chapter_outlines",
                return_value={},
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node.get_scene_character_context",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Neo4j connection failed"),
            ),
        ):
            from core.langgraph.nodes.context_retrieval_node import retrieve_context

            result = await retrieve_context(state)

        assert result["has_fatal_error"] is True
        assert result["last_error"] is not None
        assert "character profiles" in result["last_error"]
        assert result["error_node"] == "retrieve_context"

    async def test_none_chapter_plan_returns_current_node(self, tmp_path: Path) -> None:
        state: NarrativeState = {
            "project_dir": str(tmp_path),
            "current_chapter": 1,
            "current_scene_index": 0,
        }

        with (
            patch(
                "core.langgraph.nodes.context_retrieval_node.get_chapter_plan",
                return_value=None,
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node.ContentManager",
            ),
        ):
            from core.langgraph.nodes.context_retrieval_node import retrieve_context

            result = await retrieve_context(state)

        assert result["current_node"] == "retrieve_context"
