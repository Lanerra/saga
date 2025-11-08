from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def write_text_file(path: str | Path, text: str) -> None:
    """
    Write text content to a file with consistent UTF-8 encoding and LF newlines.

    Guarantees:
    - Converts ``path`` to ``Path``.
    - Ensures parent directories exist.
    - Writes with encoding="utf-8" and newline="\\n".
    - Overwrites existing file content.
    - No logging side effects.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Normalize to string and ensure LF newlines.
    data = str(text).replace("\r\n", "\n").replace("\r", "\n")

    with target.open("w", encoding="utf-8", newline="\n") as f:
        f.write(data)


def write_yaml_file(path: str | Path, data: Any) -> None:
    """
    Write YAML content to a file with consistent UTF-8 encoding and LF newlines.

    Guarantees:
    - Converts ``path`` to ``Path``.
    - Ensures parent directories exist.
    - Uses yaml.dump with:
        - default_flow_style=False
        - sort_keys=False
        - allow_unicode=True
      (callers can rely on globally-registered representers such as _LiteralString)
    - Writes with encoding="utf-8" and newline="\\n".
    - Overwrites existing file content.
    - No logging or global YAML configuration changes.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Dump YAML to a string using consistent options.
    yaml_text = yaml.dump(
        data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )

    # Normalize line endings defensively to LF.
    yaml_text = yaml_text.replace("\r\n", "\n").replace("\r", "\n")

    with target.open("w", encoding="utf-8", newline="\n") as f:
        f.write(yaml_text)
