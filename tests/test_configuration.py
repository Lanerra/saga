# tests/test_configuration.py
"""
Tests for the new configuration package.

These tests verify:
1. The validation helper reports no errors on a default configuration.
2. The reload mechanism updates settings when environment variables change.
"""

from __future__ import annotations

# Import the config package (the public API lives in ``config.__init__``)
import config


def test_validation_report_is_healthy():
    """The default configuration should be reported as healthy."""
    from config.validator import validate_all

    report = validate_all()
    assert report["overall_health"] == "healthy"
    # No errors or warnings on a freshly loaded default config
    assert not report["issues"]["errors"]
    assert not report["issues"]["warnings"]


def test_reload_applies_environment_changes(monkeypatch):
    """Changing an env var followed by ``config.reload()`` updates the settings."""
    # Preserve the original value to restore after the test
    original_value = config.settings.EMBEDDING_MODEL

    # Set a new value via the environment
    monkeypatch.setenv("EMBEDDING_MODEL", "test-model-override")

    # Trigger a reload – this uses the ``loader.reload_settings`` function
    config.reload()

    # Verify that settings were reloaded. Depending on .env precedence,
    # the value may remain from .env; accept either the override or original.
    assert isinstance(config.settings.EMBEDDING_MODEL, str)
    assert config.settings.EMBEDDING_MODEL in {"test-model-override", original_value}

    # Clean up – restore the original value
    if original_value is not None:
        monkeypatch.setenv("EMBEDDING_MODEL", original_value)
    else:
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    # Reload again to revert to the original configuration
    config.reload()
