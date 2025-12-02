import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

import config.settings as settings_mod
from config.settings import settings

print('"""Config package – central configuration for SAGA."""')
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
print('    """Return the value of a configuration key."""')
print("    return getattr(settings, key)")

print("\ndef set(key: str, value):")
print('    """Set a configuration key at runtime."""')
print("    setattr(settings, key, value)")

print("\ndef reload():")
print('    """Reload the configuration from the environment and .env file."""')
print("    from .loader import reload_settings")
print("    reload_settings()")
