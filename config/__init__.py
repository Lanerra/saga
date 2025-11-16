# config/__init__.py
"""
Config package – central configuration for SAGA.

Provides:
- `settings`: the loaded SagaSettings instance.
- `get(key)`: retrieve a configuration value.
- `set(key, value)`: update a configuration value at runtime.
- `reload()`: reload settings from the environment / .env file.

All fields from the settings are also exposed as module‑level globals for
backward compatibility (e.g. `EMBEDDING_CACHE_SIZE`).
"""

from .settings import settings as settings


def get(key: str):
    """Return the value of a configuration key."""
    return getattr(settings, key)


def set(key: str, value):
    """Set a configuration key at runtime."""
    setattr(settings, key, value)


def reload():
    """Reload the configuration from the environment and .env file."""
    from .loader import reload_settings

    reload_settings()


# Expose configuration values for legacy imports.
#
# 1) Publish all Pydantic model fields (e.g. BASE_OUTPUT_DIR, CHAPTERS_DIR as
#    raw field values).
# 2) Then, override with any UPPERCASE constants defined at module scope in
#    config.settings (these include the fully-joined output paths like
#    CHAPTERS_DIR, CHAPTER_LOGS_DIR). This ensures modules importing
#    `config.CHAPTERS_DIR` get the resolved path under BASE_OUTPUT_DIR
#    rather than the raw field value "chapters".
import importlib

# Step 1: export raw fields
for _field_name in settings.model_fields:
    globals()[_field_name] = getattr(settings, _field_name)

# Step 2: override with settings module UPPERCASE constants (joined paths, etc.)
_settings_mod = importlib.import_module(".settings", __package__)
for _name in dir(_settings_mod):
    if _name.isupper():
        globals()[_name] = getattr(_settings_mod, _name)

# Export legacy compatibility objects (Models, Temperatures) if they exist
for _obj_name in ("Models", "Temperatures"):
    if hasattr(_settings_mod, _obj_name):
        globals()[_obj_name] = getattr(_settings_mod, _obj_name)

# Export additional objects needed for backward compatibility (e.g., structlog formatters)
if hasattr(_settings_mod, "simple_formatter"):
    globals()["simple_formatter"] = _settings_mod.simple_formatter
if hasattr(_settings_mod, "rich_formatter"):
    globals()["rich_formatter"] = _settings_mod.rich_formatter
