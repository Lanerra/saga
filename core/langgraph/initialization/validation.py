# core/langgraph/initialization/validation.py
"""Validate presence of initialization artifacts on disk.

This module provides a lightweight check that expected initialization files exist under
a project directory. It does not parse or validate the file contents.
"""

from collections.abc import Iterable
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

PathLike = str | Path


def _require_file(root: Path, relative: str, missing: list[str]) -> None:
    """Append a missing-file message when `root/relative` does not exist."""
    path = root / relative
    if not path.exists():
        missing.append(f"Missing {relative}")


def _require_any(root: Path, pattern: str, description: str, missing: list[str]) -> None:
    """Append `description` when no file under `root` matches `pattern`."""
    matches: Iterable[Path] = root.glob(pattern)
    if not any(matches):
        missing.append(description)


def validate_initialization_artifacts(
    project_dir: PathLike,
) -> tuple[bool, list[str]]:
    """Validate that core initialization artifacts exist under `project_dir`.

    Args:
        project_dir: Project root directory to validate.

    Returns:
        Tuple of `(ok, missing)` where:
        - `ok` is True when all required artifacts exist.
        - `missing` is a list of human-readable messages describing missing artifacts.

    Notes:
        This is a presence-only check intended for advisory use. It does not parse or
        validate file contents.
    """
    root = Path(project_dir)

    missing: list[str] = []

    # Core config
    _require_file(root, "saga.yaml", missing)

    # Outline
    _require_file(root, "outline/structure.yaml", missing)
    _require_file(root, "outline/beats.yaml", missing)

    # Characters (at least one YAML file)
    _require_any(
        root,
        "characters/*.yaml",
        "Missing any characters/*.yaml files",
        missing,
    )

    # World files
    _require_file(root, "world/items.yaml", missing)
    _require_file(root, "world/rules.yaml", missing)
    _require_file(root, "world/history.yaml", missing)

    ok = len(missing) == 0

    # This helper is safe to call repeatedly; log at debug level only.
    if ok:
        logger.debug(
            "Initialization artifacts validation passed",
            project_dir=str(root),
        )
    else:
        logger.debug(
            "Initialization artifacts validation found missing artifacts",
            project_dir=str(root),
            missing=missing,
        )

    return ok, missing


__all__ = ["validate_initialization_artifacts"]
