# config/loader.py
"""
Configuration reload utilities for the SAGA system.

The main public function is ``reload_settings()`` which:
1. Reloads environment variables from ``.env`` (via ``dotenv.load_dotenv``).
2. Re‑creates the ``SagaSettings`` instance so that any changed values are applied.
3. Updates the symbols exported by ``config.__init__`` (the module‑level globals
   used for backward compatibility) to reflect the new values.

Optionally you can hook ``reload_settings()`` to a ``SIGHUP`` signal so that an
operator can trigger a live configuration reload without restarting the
process.
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
def _import_settings_module():
    import config.settings as _settings_mod

    return _settings_mod


def reload_settings() -> bool:
    """
    Reload configuration from the environment and refresh the ``config`` package.

    Returns ``True`` on success, ``False`` on failure.
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
        from config import (
            __init__ as config_pkg,  # pylint: disable=import-outside-toplevel
        )

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


def _handle_sighup(signum: int, frame: Any) -> None:  # pragma: no cover
    """Signal handler that invokes ``reload_settings``."""
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
