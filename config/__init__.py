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


# Expose every setting as a module‑level constant for legacy code.
# Pydantic v2 stores field definitions in `model_fields`.
import importlib

# Expose every setting as a module‑level constant for legacy code.
# Pydantic v2 stores field definitions in `model_fields`.
for _field_name in settings.model_fields:
    globals()[_field_name] = getattr(settings, _field_name)

# Export additional uppercase constants defined directly in the settings module
# (e.g., REVISION_EVALUATION_THRESHOLD, REVISION_COHERENCE_THRESHOLD, etc.).
# These are not part of the Pydantic model fields but are still required by
# legacy imports throughout the codebase.
_settings_mod = importlib.import_module(".settings", __package__)

for _name in dir(_settings_mod):
    if _name.isupper() and not hasattr(settings, _name):
        globals()[_name] = getattr(_settings_mod, _name)

# Export legacy compatibility objects (Models, Temperatures) if they exist
for _obj_name in ("Models", "Temperatures"):
    if hasattr(_settings_mod, _obj_name):
        globals()[_obj_name] = getattr(_settings_mod, _obj_name)

# Export additional objects needed for backward compatibility (e.g., structlog formatter)
if hasattr(_settings_mod, "formatter"):
    globals()["formatter"] = _settings_mod.formatter
