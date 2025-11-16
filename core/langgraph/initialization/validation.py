# core/langgraph/initialization/validation.py
from collections.abc import Iterable
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

PathLike = str | Path


def _require_file(root: Path, relative: str, missing: list[str]) -> None:
    """Check that a specific file exists under root.

    Append a human-readable message to `missing` if it does not.

    Args:
        root: Root directory path.
        relative: Relative path to the file from root.
        missing: List to append missing file messages to.
    """
    path = root / relative
    if not path.exists():
        missing.append(f"Missing {relative}")


def _require_any(
    root: Path, pattern: str, description: str, missing: list[str]
) -> None:
    """Check that at least one file matching pattern exists under root.

    Append description to missing if none found.

    Args:
        root: Root directory path.
        pattern: Glob pattern to match files (e.g., "characters/*.yaml").
        description: Human-readable description to append if no matches found.
        missing: List to append missing file messages to.
    """
    matches: Iterable[Path] = root.glob(pattern)
    if not any(matches):
        missing.append(description)


def validate_initialization_artifacts(
    project_dir: PathLike,
) -> tuple[bool, list[str]]:
    """
    Validate presence of core initialization artifacts in `project_dir`.

    This is a lightweight, non-breaking check intended for advisory use only.
    It verifies that the expected files/directories produced by the
    initialization workflow exist, without parsing or validating their content.

    Expected:
    - saga.yaml
    - outline/structure.yaml
    - outline/beats.yaml
    - at least one characters/*.yaml file
    - world/items.yaml
    - world/rules.yaml
    - world/history.yaml

    Returns:
        (ok, missing)
        ok: True if all artifacts exist, False otherwise.
        missing: List of human-readable descriptions for each missing artifact.
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
