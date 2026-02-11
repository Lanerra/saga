# tests/core/langgraph/test_state_helpers.py
"""Tests for pure state-clearing helper functions."""

from core.langgraph.state_helpers import (
    _safe_list_update,
    clear_error_state,
    clear_extraction_state,
    clear_generation_artifacts,
    clear_validation_state,
)


class TestSafeListUpdate:
    """Verify _safe_list_update produces new lists without mutation."""

    def test_returns_combined_items(self) -> None:
        """Both existing and new items appear in the result."""
        result = _safe_list_update([1, 2], [3, 4])
        assert result == [1, 2, 3, 4]

    def test_does_not_mutate_original_list(self) -> None:
        """The returned list is a distinct object from the input."""
        existing = [1, 2]
        new_items = [3]
        result = _safe_list_update(existing, new_items)

        assert result is not existing
        assert existing == [1, 2]

    def test_preserves_order(self) -> None:
        """Existing items come first, then new items, in their original order."""
        result = _safe_list_update(["a", "c"], ["b", "d"])
        assert result == ["a", "c", "b", "d"]

    def test_empty_existing_list(self) -> None:
        """An empty existing list yields only the new items."""
        result = _safe_list_update([], [10, 20])
        assert result == [10, 20]

    def test_empty_new_items(self) -> None:
        """Empty new_items yields a copy of existing."""
        existing = [5, 6]
        result = _safe_list_update(existing, [])

        assert result == [5, 6]
        assert result is not existing


class TestClearGenerationArtifacts:
    """Verify clear_generation_artifacts returns the correct reset dict."""

    def test_exact_return_value(self) -> None:
        """Returns dict with all generation fields set to their cleared values."""
        result = clear_generation_artifacts()
        assert result == {
            "draft_ref": None,
            "embedding_ref": None,
            "scene_embeddings_ref": None,
            "generated_embedding": None,
            "scene_drafts_ref": None,
            "current_scene_index": 0,
            "chapter_plan_scene_count": 0,
        }

    def test_key_count_and_types(self) -> None:
        """Contains exactly 7 keys with correct value types."""
        result = clear_generation_artifacts()
        assert len(result) == 7
        assert result["current_scene_index"] == 0
        assert isinstance(result["current_scene_index"], int)
        assert result["chapter_plan_scene_count"] == 0
        assert isinstance(result["chapter_plan_scene_count"], int)
        for key in ("draft_ref", "embedding_ref", "scene_embeddings_ref", "generated_embedding", "scene_drafts_ref"):
            assert result[key] is None

    def test_idempotent(self) -> None:
        """Consecutive calls produce equal results."""
        assert clear_generation_artifacts() == clear_generation_artifacts()


class TestClearValidationState:
    """Verify clear_validation_state returns the correct reset dict."""

    def test_exact_return_value(self) -> None:
        """Returns dict with all validation fields set to their cleared values."""
        result = clear_validation_state()
        assert result == {
            "contradictions": [],
            "needs_revision": False,
            "revision_guidance_ref": None,
        }

    def test_key_count_and_types(self) -> None:
        """Contains exactly 3 keys with correct value types."""
        result = clear_validation_state()
        assert len(result) == 3
        assert isinstance(result["contradictions"], list)
        assert result["contradictions"] == []
        assert result["needs_revision"] is False
        assert result["revision_guidance_ref"] is None

    def test_idempotent(self) -> None:
        """Consecutive calls produce equal results."""
        assert clear_validation_state() == clear_validation_state()


class TestClearErrorState:
    """Verify clear_error_state returns the correct reset dict."""

    def test_exact_return_value(self) -> None:
        """Returns dict with all error fields set to their cleared values."""
        result = clear_error_state()
        assert result == {
            "last_error": None,
            "has_fatal_error": False,
            "error_node": None,
        }

    def test_key_count_and_types(self) -> None:
        """Contains exactly 3 keys with correct value types."""
        result = clear_error_state()
        assert len(result) == 3
        assert result["last_error"] is None
        assert result["has_fatal_error"] is False
        assert isinstance(result["has_fatal_error"], bool)
        assert result["error_node"] is None

    def test_idempotent(self) -> None:
        """Consecutive calls produce equal results."""
        assert clear_error_state() == clear_error_state()


class TestClearExtractionState:
    """Verify clear_extraction_state returns the correct reset dict."""

    def test_exact_return_value(self) -> None:
        """Returns dict with all extraction fields set to their cleared values."""
        result = clear_extraction_state()
        assert result == {
            "extracted_entities_ref": None,
            "extracted_relationships_ref": None,
        }

    def test_key_count_and_types(self) -> None:
        """Contains exactly 2 keys, both None."""
        result = clear_extraction_state()
        assert len(result) == 2
        assert result["extracted_entities_ref"] is None
        assert result["extracted_relationships_ref"] is None

    def test_idempotent(self) -> None:
        """Consecutive calls produce equal results."""
        assert clear_extraction_state() == clear_extraction_state()
