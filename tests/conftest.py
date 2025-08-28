# tests/conftest.py
import os
import sys

# Ensure repository root is on PYTHONPATH for tests
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
