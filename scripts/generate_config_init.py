"""Regenerate [`config.__init__`](config/__init__.py:1) from [`config.settings`](config/settings.py:1).

This script prints Python source code to stdout. It is intended to be run from the repo
root, then redirected into `config/__init__.py`.

Side effects:
- Reads the active process environment and `.env` indirectly by importing
  [`config.settings`](config/settings.py:1).
- Writes generated source code to stdout.

Notes:
    This script uses `sys.path.insert(0, os.getcwd())` to ensure imports resolve when run
    directly. It assumes the working directory is the project root.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

import config.settings as settings_mod
from config.settings import settings

print(
    '"""Expose SAGA configuration as stable module-level constants.\n\n'
    "This package provides a backwards-compatible facade over the underlying Pydantic settings\n"
    "model defined in `config.settings`. The primary API is the `settings` singleton plus a set\n"
    "of module-level constants mirroring its fields.\n\n"
    "Configuration precedence and lifecycle:\n"
    "- On initial import, configuration is loaded by importing `config.settings`,\n"
    "  which constructs the `settings` singleton.\n"
    "- Values come from the process environment and may be sourced from a `.env` file.\n"
    "- `reload()` triggers a refresh that re-reads `.env` with override enabled,\n"
    "  then replaces this module's exported values.\n\n"
    "Notes:\n"
    "    This module intentionally duplicates values into module globals for legacy callers.\n"
    '"""\n'
)
print("from .settings import settings")
print("from .settings import (")
# Add specific objects
objects = ["Models", "Temperatures", "simple_formatter", "rich_formatter"]
for obj in objects:
    print(f"    {obj},")
print(")")

print("\n# Explicit exports for MyPy compatibility")

seen = set()

# First, module level constants (these include the overridden paths)
for name in dir(settings_mod):
    if name.isupper():
        print(f"{name} = settings_mod.{name}")
        seen.add(name)

# Then remaining fields from settings
# Use model_fields from the class to avoid deprecation warning if possible, but instance is fine for now
for field_name in settings.model_fields:
    if field_name not in seen:
        print(f"{field_name} = settings.{field_name}")

print("\n# Legacy compatibility functions")
print("def get(key: str):")
print(
    '    """Return the value of a configuration attribute from `settings`.\n\n'
    "    Args:\n"
    "        key: Attribute name on the `settings` singleton.\n\n"
    "    Returns:\n"
    "        The current value of the named attribute.\n\n"
    "    Raises:\n"
    "        AttributeError: If `key` is not a valid attribute on `settings`.\n"
    '    """'
)
print("    return getattr(settings, key)")

print("\ndef set(key: str, value):")
print(
    '    """Set a configuration attribute on `settings` at runtime.\n\n'
    "    This mutates the in-memory settings instance and does not persist to `.env`.\n\n"
    "    Args:\n"
    "        key: Attribute name on the `settings` singleton.\n"
    "        value: Value to assign.\n\n"
    "    Raises:\n"
    "        AttributeError: If `key` is not a valid attribute on `settings`.\n"
    '    """'
)
print("    setattr(settings, key, value)")

print("\ndef reload():")
print(
    '    """Reload configuration and refresh this package\'s exported constants.\n\n'
    "    This delegates to `config.loader.reload_settings()`, which may overwrite process\n"
    "    environment variables by re-reading `.env` with override enabled.\n"
    '    """'
)
print("    from .loader import reload_settings")
print("    reload_settings()")
