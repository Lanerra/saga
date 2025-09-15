# tests/test_integration_smoke.py
"""Smoke test to verify core integration is working."""

import pytest

from agents.knowledge_agent import KnowledgeAgent
from processing.state_tracker import StateTracker


@pytest.mark.asyncio
async def test_basic_state_tracker_functionality():
    """Basic smoke test for StateTracker functionality."""
    tracker = StateTracker()

    # Test basic reservation
    result = await tracker.reserve("TestCharacter", "character", "A test character")
    assert result is True

    # Test duplicate prevention
    result2 = await tracker.reserve("TestCharacter", "character", "Duplicate character")
    assert result2 is False

    # Test metadata retrieval
    metadata = await tracker.check("TestCharacter")
    assert metadata is not None
    assert metadata["name"] == "TestCharacter"
    assert metadata["type"] == "character"


@pytest.mark.asyncio
async def test_knowledge_agent_initialization():
    """Test that KnowledgeAgent initializes correctly."""
    agent = KnowledgeAgent()
    assert agent is not None

    # Test that methods exist
    assert hasattr(agent, "prioritize_candidates_with_state_tracker")
    assert hasattr(agent, "prioritize_candidates_with_state_tracker_async")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
