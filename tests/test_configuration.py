# tests/test_configuration.py
"""
Tests for the new configuration package.

These tests verify:
1. The validation helper reports no errors on a default configuration.
2. The reload mechanism updates settings when environment variables change.
"""

from __future__ import annotations

# Import the config package (the public API lives in ``config.__init__``)
import pytest

import config


def test_validation_report_is_healthy() -> None:
    """The default configuration should be reported as healthy."""
    from config.validator import validate_all

    report = validate_all()
    assert report["overall_health"] == "healthy"
    # No errors or warnings on a freshly loaded default config
    assert not report["issues"]["errors"]
    assert not report["issues"]["warnings"]


@pytest.mark.unbound_settings
def test_reload_applies_environment_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing an env var followed by ``config.reload()`` updates the settings."""
    try:
        with monkeypatch.context() as environment:
            environment.setenv("EMBEDDING_MODEL", "test-model-override")
            assert config.reload(env_file=None) is True
            assert config.settings.EMBEDDING_MODEL == "test-model-override"
            assert config.snapshot_settings().EMBEDDING_MODEL == "test-model-override"
    finally:
        config.reload(env_file=None)
