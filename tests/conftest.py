# tests/conftest.py
import os
import sys

import pytest

# Ensure repository root is on PYTHONPATH for tests
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom CLI options for this repo.

    --unit-stubs: accept the flag and optionally adjust collection behavior to
    focus on lightweight, hermetic tests. This flag is a no-op by default but
    prevents failures from unknown options and allows CI toggling.
    """
    parser.addoption(
        "--unit-stubs",
        action="store_true",
        default=False,
        help="Run unit tests with stubs/mocks; ignore heavier suites.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """When --unit-stubs is passed, skip heavier-marked tests by default.

    We respect existing markers defined in pyproject.toml: integration, slow,
    performance, ui, websocket, checkpoint, fault_injection, load_test.
    """
    if not config.getoption("--unit-stubs"):
        return

    skip_marker = pytest.mark.skip(reason="skipped by --unit-stubs")
    heavy_markers = {
        "integration",
        "slow",
        "performance",
        "ui",
        "websocket",
        "checkpoint",
        "fault_injection",
        "load_test",
    }
    for item in items:
        for m in item.iter_markers():
            if m.name in heavy_markers:
                item.add_marker(skip_marker)
                break
