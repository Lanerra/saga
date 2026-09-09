"""Tests for core/langgraph/initialization/outline_relationships_node.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.langgraph.initialization.outline_relationships_node import (
    _parse_relationships_extraction,
    extract_outline_relationships,
)
from core.langgraph.state import NarrativeState
from tests.fakes.service_context import patch_service
from tests.test_staged_initialization import example_state, with_catalog
from utils.text_processing import generate_entity_id


def _make_state(tmp_path: str) -> NarrativeState:
    return {
        "project_dir": tmp_path,
        "title": "The Lost Kingdom",
        "genre": "Fantasy",
        "protagonist_name": "Aria",
        "setting": "Medieval realm",
    }


EXAMPLE_LLM_RESPONSE = json.dumps(
    {
        "kg_triples": [
            {
                "subject": "Aria",
                "predicate": "LIVES_IN",
                "object_entity": "The Kingdom",
                "description": "Aria resides in The Kingdom",
            },
            {
                "subject": "Aria",
                "predicate": "FRIEND_OF",
                "object_entity": "Borin",
                "description": "They are childhood friends",
            },
        ]
    }
)


class TestParseRelationshipsExtraction:
    """Tests for the _parse_relationships_extraction helper."""

    def test_valid_json_with_triples(self) -> None:
        response = json.dumps(
            {
                "kg_triples": [
                    {
                        "subject": "Alice",
                        "predicate": "KNOWS",
                        "object_entity": "Bob",
                        "description": "They met at the market",
                    },
                ]
            }
        )

        result = _parse_relationships_extraction(response)

        assert len(result) == 1
        assert result[0] == {
            "source_name": "Alice",
            "target_name": "Bob",
            "relationship_type": "KNOWS",
            "description": "They met at the market",
            "chapter": 0,
            "confidence": 0.8,
        }

    def test_code_fenced_json(self) -> None:
        inner = json.dumps(
            {
                "kg_triples": [
                    {
                        "subject": "X",
                        "predicate": "RELATED_TO",
                        "object_entity": "Y",
                        "description": "",
                    },
                ]
            }
        )
        response = f"```json\n{inner}\n```"

        result = _parse_relationships_extraction(response)

        assert len(result) == 1
        assert result[0]["source_name"] == "X"
        assert result[0]["target_name"] == "Y"

    def test_empty_kg_triples_returns_empty_list(self) -> None:
        response = json.dumps({"kg_triples": []})
        result = _parse_relationships_extraction(response)
        assert result == []

    def test_rejects_incomplete_triples(self) -> None:
        response = json.dumps(
            {
                "kg_triples": [
                    {"subject": "A", "predicate": "", "object_entity": "B", "description": ""},
                    {"subject": "C", "predicate": "KNOWS", "object_entity": "D", "description": "ok"},
                ]
            }
        )

        with pytest.raises(ValueError, match="Relationship triple has an incomplete identity"):
            _parse_relationships_extraction(response)

    def test_handles_dict_subject_and_object(self) -> None:
        response = json.dumps(
            {
                "kg_triples": [
                    {
                        "subject": {"name": "Alice"},
                        "predicate": "LOVES",
                        "object_entity": {"name": "Bob"},
                        "description": "A romance",
                    },
                ]
            }
        )

        result = _parse_relationships_extraction(response)

        assert len(result) == 1
        assert result[0]["source_name"] == "Alice"
        assert result[0]["target_name"] == "Bob"

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _parse_relationships_extraction("not json at all")

    def test_non_dict_top_level_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected JSON object"):
            _parse_relationships_extraction(json.dumps([1, 2, 3]))

    def test_kg_triples_not_a_list_raises(self) -> None:
        with pytest.raises(ValueError, match="kg_triples must be a JSON array"):
            _parse_relationships_extraction(json.dumps({"kg_triples": "bad"}))

    def test_strips_whitespace_from_names(self) -> None:
        response = json.dumps(
            {
                "kg_triples": [
                    {
                        "subject": "  Alice  ",
                        "predicate": "  KNOWS  ",
                        "object_entity": "  Bob  ",
                        "description": "  desc  ",
                    },
                ]
            }
        )

        result = _parse_relationships_extraction(response)

        assert result[0]["source_name"] == "Alice"
        assert result[0]["target_name"] == "Bob"
        assert result[0]["relationship_type"] == "KNOWS"
        assert result[0]["description"] == "desc"

    def test_multiple_triples_preserved_in_order(self) -> None:
        response = json.dumps(
            {
                "kg_triples": [
                    {"subject": "A", "predicate": "R1", "object_entity": "B", "description": ""},
                    {"subject": "C", "predicate": "R2", "object_entity": "D", "description": ""},
                    {"subject": "E", "predicate": "R3", "object_entity": "F", "description": ""},
                ]
            }
        )

        result = _parse_relationships_extraction(response)

        assert len(result) == 3
        assert [r["source_name"] for r in result] == ["A", "C", "E"]


class TestExtractOutlineRelationships:
    """Tests for the extract_outline_relationships node."""

    async def test_no_catalog_fails_before_extraction(self, tmp_path: Path) -> None:
        state = _make_state(str(tmp_path))
        with patch_service('language_model') as provider:
            provider.async_call_llm = AsyncMock()
            with pytest.raises(ValueError, match="selected initialization_catalog_ref"):
                await extract_outline_relationships(state)
            provider.async_call_llm.assert_not_awaited()

    async def test_successful_extraction(self, tmp_path: Path) -> None:
        state = with_catalog(example_state(tmp_path))
        state["outline_relationships_ref"] = None
        response = json.dumps({"kg_triples": [{
            "source_id": generate_entity_id("Ada", "character"), "source_label": "Character",
            "target_id": generate_entity_id("Ada", "character"), "target_label": "Character",
            "relationship_type": "FRIEND_OF", "description": "Synthetic identity assertion.",
        }]})
        with patch_service('language_model') as fake_llm:
            fake_llm.async_call_llm = AsyncMock(return_value=(response, {"prompt_tokens": 100, "completion_tokens": 50}))
            result = await extract_outline_relationships(state)

        assert result["current_node"] == "outline_relationships"
        assert result["initialization_step"] == "outline_relationships_extracted"
        assert result["outline_relationships_ref"] is not None
        ref = result["outline_relationships_ref"]
        assert ref["content_type"] == "outline_relationships"

    async def test_parse_failure_raises_without_repair_loop(self, tmp_path: Path) -> None:
        state = with_catalog(example_state(tmp_path))
        state["outline_relationships_ref"] = None
        before = {path: path.read_bytes() for path in (tmp_path / ".saga/content/outline_relationships").iterdir()}
        with patch_service('language_model') as fake_llm:
            fake_llm.async_call_llm = AsyncMock(return_value=("this is not json", {"prompt_tokens": 50, "completion_tokens": 20}))
            with pytest.raises(json.JSONDecodeError):
                await extract_outline_relationships(state)
            assert fake_llm.async_call_llm.await_count == 1
        assert state["outline_relationships_ref"] is None
        assert {path: path.read_bytes() for path in (tmp_path / ".saga/content/outline_relationships").iterdir()} == before
