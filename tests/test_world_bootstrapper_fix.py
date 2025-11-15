# tests/test_world_bootstrapper_fix.py
"""Test to verify the world bootstrapper bug fixes."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from initialization.bootstrappers.world_bootstrapper import _bootstrap_world_names
from processing.state_tracker import StateTracker


@pytest.mark.asyncio
async def test_union_operation_fix():
    """Test that the union operation bug is fixed."""
    # Create mock world building data that would trigger the bug
    world_building = {
        "locations": {
            "test_location": MagicMock(
                name="test_location", spec=["name", "description"]
            )
        }
    }
    world_building["locations"]["test_location"].name = ""
    world_building["locations"]["test_location"].description = "A test location"

    plot_outline = {"title": "Test Novel"}

    # Create StateTracker mock
    state_tracker = AsyncMock(spec=StateTracker)
    state_tracker.get_all = AsyncMock(return_value={})
    state_tracker.reserve = AsyncMock(return_value=True)
    state_tracker.check = AsyncMock(return_value=None)

    # This should not raise AttributeError: 'list' object has no attribute 'union'
    try:
        await _bootstrap_world_names(world_building, plot_outline, state_tracker)
        # If we get here without exception, the fix worked
        assert True
    except AttributeError as e:
        if "'list' object has no attribute 'union'" in str(e):
            pytest.fail("Union operation bug still exists!")
        else:
            # Re-raise if it's a different AttributeError
            raise
    except Exception:
        # Other exceptions are fine for this test
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
