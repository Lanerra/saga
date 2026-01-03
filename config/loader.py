# config/loader.py
"""Reload SAGA configuration at runtime.

The primary entry point is [`reload_settings()`](config/loader.py:35). It refreshes
configuration by:

1. Re-reading `.env` into the process environment using `dotenv.load_dotenv()` with
   override enabled.
2. Reloading [`config.settings`](config/settings.py:1) to construct a fresh
   [`settings`](config/settings.py:356) singleton.
3. Replacing the `settings` object and module-level constants exported by
   [`config`](config/__init__.py:1) for legacy callers.

Configuration precedence:
- This loader explicitly uses `load_dotenv(override=True)`. That means `.env` values can
  overwrite environment variables that were already present in the process. This differs
  from the default import-time behavior in [`config.settings`](config/settings.py:1).

Side effects:
- Mutates the process environment by loading `.env`.
- Mutates module state in [`config`](config/__init__.py:1) by rebinding `settings` and
  updating module globals.
- When SIGHUP handling is enabled, importing this module registers a signal handler.
"""

from __future__ import annotations

import importlib
import os
import signal
from typing import Any

from dotenv import load_dotenv


# The settings module contains the ``SagaSettings`` class and the ``settings``
# singleton instance.  Import it lazily inside the function so that reloading
# works correctly when the function is called multiple times.
def _import_settings_module() -> Any:
    """Import and return the `config.settings` module.

    Returns:
        The imported module object for [`config.settings`](config/settings.py:1).

    Notes:
        This indirection exists so [`reload_settings()`](config/loader.py:35) can reload
        the settings module repeatedly without relying on module-level caching behavior.
    """
    import config.settings as _settings_mod

    return _settings_mod


def reload_settings() -> bool:
    """Reload configuration from `.env` and refresh the `config` package exports.

    This function is intentionally non-throwing: it returns a boolean status and
    suppresses all exceptions. Failures are treated as internal errors (unexpected
    runtime conditions during reload), not user configuration errors.

    Returns:
        `True` when reload completes and `config` globals have been refreshed, otherwise
        `False`.

    Notes:
        This function uses `load_dotenv(override=True)`, which can overwrite existing
        process environment variables.
    """
    try:
        # 1️⃣ Reload .env – ``override=True`` forces a refresh of existing keys.
        load_dotenv(override=True)

        # 2️⃣ Reload the settings module – this re‑executes the ``SagaSettings``
        #    definition and creates a fresh ``settings`` instance.
        settings_mod = _import_settings_module()
        importlib.reload(settings_mod)

        # 3️⃣ Update the ``config`` package globals.
        #    ``config.__init__`` imports ``settings`` as a name, so we need to
        #    replace that reference with the newly created instance.
        import config as config_pkg  # pylint: disable=import-outside-toplevel

        # Replace the ``settings`` object.
        config_pkg.settings = settings_mod.settings

        # Re‑expose every field as a module‑level constant for legacy code.
        # ``settings.model_fields`` gives us all field names.
        for field_name in settings_mod.settings.model_fields:
            setattr(config_pkg, field_name, getattr(settings_mod.settings, field_name))

        return True
    except Exception:  # pragma: no cover – defensive fallback
        # In a real system you would log this; for now we simply return False.
        return False


# -------------------------------------------------------------------------
# Optional signal handling – allow ``kill -HUP <pid>`` to trigger a reload.
# This is a convenience for developers and can be disabled in production
# by not importing this module.
# -------------------------------------------------------------------------


def _handle_sighup(signum: int, _frame: Any) -> None:  # pragma: no cover
    """Handle `SIGHUP` by reloading configuration.

    Args:
        signum: Signal number (unused but required by signal handler signature).
        _frame: Current stack frame (unused but required by signal handler signature).

    Side effects:
        Writes a success/failure message to stdout.
    """
    success = reload_settings()
    if success:
        print("[config] Configuration reloaded via SIGHUP")
    else:
        print("[config] Failed to reload configuration via SIGHUP")


# Register the handler when the module is imported.
# If the environment variable ``CONFIG_DISABLE_SIGHUP`` is set to any value,
# we skip registration (useful for containers where signals are managed externally).
if not os.getenv("CONFIG_DISABLE_SIGHUP"):
    signal.signal(signal.SIGHUP, _handle_sighup)
