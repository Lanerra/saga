"""Tests for core/langgraph/nodes/context_retrieval_node.py.

Focuses on pure functions that do not require external services, plus
a main-flow integration test with patched dependencies.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from core.langgraph.nodes.context_retrieval_node import (
    _extract_scene_characters,
    _smart_truncate_scene,
)


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
        scene = {"characters": []}
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
            "core.langgraph.nodes.context_retrieval_node.count_tokens",
            return_value=5,
        ):
            result = _smart_truncate_scene(text, "test-model", max_tokens=100)
        assert result == "A short scene."

    def test_long_text_contains_head_and_tail(self) -> None:
        words = [f"word{i}" for i in range(200)]
        text = " ".join(words)

        with patch(
            "core.langgraph.nodes.context_retrieval_node.count_tokens",
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
            "core.langgraph.nodes.context_retrieval_node.count_tokens",
            return_value=1000,
        ):
            result = _smart_truncate_scene(text, "test-model", max_tokens=100)

        parts = result.split("[...]")
        assert len(parts) == 2
        head_words = parts[0].strip().split()
        tail_words = parts[1].strip().split()
        assert len(head_words) == 10
        assert len(tail_words) == 80
        assert head_words[0] == "w0"

    def test_exact_budget_returns_unchanged(self) -> None:
        text = "exactly on budget text"
        with patch(
            "core.langgraph.nodes.context_retrieval_node.count_tokens",
            return_value=50,
        ):
            result = _smart_truncate_scene(text, "test-model", max_tokens=50)
        assert result == "exactly on budget text"

    def test_head_portion_is_smaller_than_tail(self) -> None:
        words = [f"w{i}" for i in range(500)]
        text = " ".join(words)

        with patch(
            "core.langgraph.nodes.context_retrieval_node.count_tokens",
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

    async def test_invalid_scene_index_returns_current_node(self, tmp_path) -> None:
        state = {
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

    async def test_successful_context_build(self, tmp_path) -> None:
        state = {
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
                "core.langgraph.nodes.context_retrieval_node._get_scene_character_context",
                new_callable=AsyncMock,
                return_value="**Scene Character Profiles:**\nAlice: A brave warrior",
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node._get_scene_specific_kg_facts",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node._get_scene_events_context",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node._get_character_relationships_context",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node._get_character_items_context",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node._get_scene_items_context",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node._get_scene_location_context",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "core.langgraph.nodes.context_retrieval_node._get_semantic_context",
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

    async def test_character_context_error_sets_fatal(self, tmp_path) -> None:
        state = {
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
                "core.langgraph.nodes.context_retrieval_node._get_scene_character_context",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Neo4j connection failed"),
            ),
        ):
            from core.langgraph.nodes.context_retrieval_node import retrieve_context

            result = await retrieve_context(state)

        assert result["has_fatal_error"] is True
        assert "character profiles" in result["last_error"]
        assert result["error_node"] == "retrieve_context"

    async def test_none_chapter_plan_returns_current_node(self, tmp_path) -> None:
        state = {
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
