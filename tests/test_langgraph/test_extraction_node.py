"""
Tests for LangGraph extraction node (Step 1.1.2).

Tests the extract_entities node and its helper functions.
"""

from unittest.mock import patch

import pytest

from core.langgraph.nodes.extraction_node import (
    _clean_llm_json,
    _map_category_to_type,
    _parse_extraction_json,
    extract_entities,
)


class TestCleanLLMJSON:
    """Tests for _clean_llm_json helper function."""

    def test_clean_markdown_code_blocks(self):
        """Test removing markdown code blocks."""
        raw = """```json
        {"test": "value"}
        ```"""
        cleaned = _clean_llm_json(raw)
        assert cleaned == '{"test": "value"}'

    def test_clean_trailing_commas(self):
        """Test removing trailing commas before closing brackets."""
        raw = '{"array": [1, 2, 3,], "nested": {"key": "value",}}'
        cleaned = _clean_llm_json(raw)
        # Check that trailing commas are removed
        assert not cleaned.endswith(",]")
        assert not cleaned.endswith(",}")
        assert cleaned == '{"array": [1, 2, 3], "nested": {"key": "value"}}'

    def test_clean_curly_quotes(self):
        """Test converting curly quotes to straight quotes."""
        # Use actual Unicode curly quotes (U+201C and U+201D)
        raw = "\u007b\u201ckey\u201d: \u201cvalue\u201d\u007d"  # {"key": "value"} with curly quotes
        cleaned = _clean_llm_json(raw)
        # Check that straight quotes are present and curly quotes are removed
        assert '"' in cleaned
        assert "\u201c" not in cleaned  # Left double quotation mark
        assert "\u201d" not in cleaned  # Right double quotation mark

    def test_clean_already_clean_json(self):
        """Test that already clean JSON is unchanged."""
        raw = '{"test": "value"}'
        cleaned = _clean_llm_json(raw)
        assert cleaned == raw

    def test_clean_with_multiple_issues(self):
        """Test cleaning JSON with multiple issues."""
        raw = """```json
        {"array": [1, 2,], "key": "value",}
        ```"""
        cleaned = _clean_llm_json(raw)
        # Check markdown blocks are removed
        assert not cleaned.startswith("```")
        # Check trailing commas are removed
        assert ",]" not in cleaned
        assert ",}" not in cleaned


class TestMapCategoryToType:
    """Tests for _map_category_to_type helper function."""

    def test_map_location_categories(self):
        """Test mapping location-related categories."""
        assert _map_category_to_type("location") == "location"
        assert _map_category_to_type("place") == "location"
        assert _map_category_to_type("settlement") == "location"
        assert _map_category_to_type("region") == "location"
        assert _map_category_to_type("structure") == "location"

    def test_map_event_categories(self):
        """Test mapping event-related categories."""
        assert _map_category_to_type("event") == "event"
        assert _map_category_to_type("ceremony") == "event"
        assert _map_category_to_type("battle") == "event"
        assert _map_category_to_type("incident") == "event"

    def test_map_character_categories(self):
        """Test mapping character-related categories."""
        assert _map_category_to_type("character") == "character"
        assert _map_category_to_type("person") == "character"
        assert _map_category_to_type("creature") == "character"

    def test_map_default_to_object(self):
        """Test that unknown categories default to object."""
        assert _map_category_to_type("artifact") == "object"
        assert _map_category_to_type("item") == "object"
        assert _map_category_to_type("weapon") == "object"
        assert _map_category_to_type("unknown_category") == "object"

    def test_map_case_insensitive(self):
        """Test that category mapping is case insensitive."""
        assert _map_category_to_type("LOCATION") == "location"
        assert _map_category_to_type("Location") == "location"
        assert _map_category_to_type("EVENT") == "event"


@pytest.mark.asyncio
class TestParseExtractionJSON:
    """Tests for _parse_extraction_json function."""

    async def test_parse_valid_json(self, sample_llm_extraction_response):
        """Test parsing valid extraction JSON."""
        result = await _parse_extraction_json(sample_llm_extraction_response, 1)

        assert result is not None
        assert "character_updates" in result
        assert "world_updates" in result
        assert "kg_triples" in result

    async def test_parse_empty_text(self):
        """Test parsing empty text returns None."""
        result = await _parse_extraction_json("", 1)
        assert result is None

    async def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns minimal structure."""
        invalid_json = '{"incomplete": '
        result = await _parse_extraction_json(invalid_json, 1)

        # Should return minimal structure as fallback
        assert result is not None
        assert "character_updates" in result
        assert "world_updates" in result
        assert "kg_triples" in result

    async def test_parse_json_with_markdown(self, sample_llm_extraction_response):
        """Test parsing JSON wrapped in markdown."""
        wrapped = f"```json\n{sample_llm_extraction_response}\n```"
        result = await _parse_extraction_json(wrapped, 1)

        assert result is not None
        assert "character_updates" in result


@pytest.mark.asyncio
class TestExtractEntities:
    """Tests for extract_entities node function."""

    async def test_extract_entities_with_no_draft_text(
        self, sample_initial_state, mock_llm_service
    ):
        """Test extraction with no draft text."""
        state = sample_initial_state
        state["draft_text"] = None

        result = await extract_entities(state)

        assert result["extracted_entities"] == {}
        assert result["extracted_relationships"] == []
        assert result["current_node"] == "extract_entities"

    async def test_extract_entities_successful(
        self,
        sample_state_with_extraction,
        mock_llm_service,
        sample_llm_extraction_response,
    ):
        """Test successful entity extraction."""
        state = sample_state_with_extraction
        state["draft_text"] = "Alice and Bob traveled through the Dark Forest..."

        # Mock LLM response
        mock_llm_service.async_call_llm.return_value = (
            sample_llm_extraction_response,
            {"prompt_tokens": 100, "completion_tokens": 50},
        )

        with patch(
            "core.langgraph.nodes.extraction_node.llm_service", mock_llm_service
        ):
            result = await extract_entities(state)

            assert "extracted_entities" in result
            assert "extracted_relationships" in result
            assert result["current_node"] == "extract_entities"
            assert result["last_error"] is None

    async def test_extract_entities_llm_failure(
        self, sample_initial_state, mock_llm_service
    ):
        """Test extraction when LLM call fails."""
        state = sample_initial_state
        state["draft_text"] = "Some text..."

        # Mock LLM to raise exception
        mock_llm_service.async_call_llm.side_effect = Exception("LLM service error")

        with patch(
            "core.langgraph.nodes.extraction_node.llm_service", mock_llm_service
        ):
            result = await extract_entities(state)

            assert result["extracted_entities"] == {}
            assert result["extracted_relationships"] == []
            assert result["last_error"] is not None

    async def test_extract_entities_empty_llm_response(
        self, sample_initial_state, mock_llm_service
    ):
        """Test extraction when LLM returns empty response."""
        state = sample_initial_state
        state["draft_text"] = "Some text..."

        # Mock LLM to return empty string
        mock_llm_service.async_call_llm.return_value = ("", None)

        with patch(
            "core.langgraph.nodes.extraction_node.llm_service", mock_llm_service
        ):
            result = await extract_entities(state)

            assert result["extracted_entities"] == {}
            assert result["extracted_relationships"] == []

    async def test_extract_entities_updates_state(
        self, sample_initial_state, mock_llm_service, sample_llm_extraction_response
    ):
        """Test that extraction properly updates state."""
        state = sample_initial_state
        state["draft_text"] = "Alice and Bob..."
        state["current_chapter"] = 5

        mock_llm_service.async_call_llm.return_value = (
            sample_llm_extraction_response,
            {"prompt_tokens": 100, "completion_tokens": 50},
        )

        with patch(
            "core.langgraph.nodes.extraction_node.llm_service", mock_llm_service
        ):
            result = await extract_entities(state)

            # State should preserve original fields
            assert result["current_chapter"] == 5
            assert result["draft_text"] == "Alice and Bob..."

            # State should have new extraction results
            assert "extracted_entities" in result
            assert "extracted_relationships" in result

    async def test_extract_entities_with_malformed_json(
        self, sample_initial_state, mock_llm_service
    ):
        """Test extraction with malformed JSON response."""
        state = sample_initial_state
        state["draft_text"] = "Some text..."

        # Mock LLM to return malformed JSON
        mock_llm_service.async_call_llm.return_value = (
            '{"character_updates": {"incomplete',
            {"prompt_tokens": 100, "completion_tokens": 50},
        )

        with patch(
            "core.langgraph.nodes.extraction_node.llm_service", mock_llm_service
        ):
            result = await extract_entities(state)

            # Should handle gracefully
            assert "extracted_entities" in result
            assert result["last_error"] is None  # Parsing errors are handled

    async def test_extract_entities_calls_llm_with_correct_params(
        self, sample_initial_state, mock_llm_service, sample_llm_extraction_response
    ):
        """Test that extraction calls LLM with correct parameters."""
        state = sample_initial_state
        state["draft_text"] = "Test chapter text..."
        state["current_chapter"] = 3
        state["extraction_model"] = "test-extraction-model"

        mock_llm_service.async_call_llm.return_value = (
            sample_llm_extraction_response,
            {"prompt_tokens": 100, "completion_tokens": 50},
        )

        with patch(
            "core.langgraph.nodes.extraction_node.llm_service", mock_llm_service
        ):
            await extract_entities(state)

            # Verify LLM was called
            assert mock_llm_service.async_call_llm.called

            # Get call args
            call_args = mock_llm_service.async_call_llm.call_args

            # Check model name
            assert call_args.kwargs.get("model_name") == "test-extraction-model"
