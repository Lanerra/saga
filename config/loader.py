# config/loader.py
"""Transactionally load defaults for future runs with process-over-file precedence."""

from __future__ import annotations

from pathlib import Path

from . import settings_mod
from .settings import SagaSettings


def reload_settings(*, env_file: Path | None = Path(".env")) -> bool:
    """Publish only a fully validated replacement; failures propagate without mutation."""
    replacement = SagaSettings(_env_file=env_file)
    import config

    settings_mod.settings = replacement
    for name in SagaSettings.model_fields:
        vars(config).pop(name, None)
    return True
