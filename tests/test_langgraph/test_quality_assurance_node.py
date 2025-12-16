# tests/test_langgraph/test_quality_assurance_node.py
"""
Unit tests for QualityAssuranceNode behavior.

Scope: LANGGRAPH-027 remediation — ensure QA history uses a real ISO-8601 timestamp
(not the placeholder "now") and that it is parseable.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

import config
from core.langgraph.nodes.quality_assurance_node import check_quality
from core.langgraph.state import NarrativeState


@pytest.mark.asyncio
async def test_check_quality_appends_parseable_iso8601_timestamp(
    sample_initial_state: NarrativeState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    check_quality() should append a QA history item with a real ISO-8601 timestamp.

    We validate parseability (datetime.fromisoformat) and timezone-awareness rather
    than exact value, to keep the test deterministic.
    """
    # Ensure QA runs for this test (avoid frequency gating).
    monkeypatch.setattr(config.settings, "ENABLE_QA_CHECKS", True)
    monkeypatch.setattr(config.settings, "QA_CHECK_FREQUENCY", 1)

    # Enable all sub-checks, but patch the I/O functions so the test remains hermetic.
    monkeypatch.setattr(config.settings, "QA_CHECK_CONTRADICTORY_TRAITS", True)
    monkeypatch.setattr(config.settings, "QA_CHECK_POST_MORTEM_ACTIVITY", True)
    monkeypatch.setattr(config.settings, "QA_DEDUPLICATE_RELATIONSHIPS", True)
    monkeypatch.setattr(config.settings, "QA_CONSOLIDATE_RELATIONSHIPS", True)

    monkeypatch.setattr(
        "core.langgraph.nodes.quality_assurance_node.find_contradictory_trait_characters",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        "core.langgraph.nodes.quality_assurance_node.find_post_mortem_activity",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        "core.langgraph.nodes.quality_assurance_node.deduplicate_relationships",
        AsyncMock(return_value=0),
    )
    monkeypatch.setattr(
        "core.langgraph.nodes.quality_assurance_node.consolidate_similar_relationships",
        AsyncMock(return_value=0),
    )

    # Minimal state: make sure QA actually runs and has a history container.
    state: NarrativeState = {**sample_initial_state}
    state["current_chapter"] = 1
    state["last_qa_chapter"] = 0
    state["qa_history"] = []

    result = await check_quality(state)

    assert isinstance(result.get("qa_history"), list)
    assert len(result["qa_history"]) == 1

    history_item = result["qa_history"][0]
    assert history_item["chapter"] == 1

    timestamp = history_item["timestamp"]
    assert isinstance(timestamp, str)
    assert timestamp != "now"

    parsed = datetime.fromisoformat(timestamp)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None
