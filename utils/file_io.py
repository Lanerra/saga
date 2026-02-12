# utils/file_io.py
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def _atomic_write(target: Path, data: str) -> None:
    """Write data to a file atomically via temp file + fsync + rename."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        os.unlink(tmp_path)
        raise


def write_text_file(path: str | Path, text: str) -> None:
    """
    Write text content to a file atomically with consistent UTF-8 encoding and LF newlines.

    Guarantees:
    - Converts ``path`` to ``Path``.
    - Ensures parent directories exist.
    - Writes with encoding="utf-8" and newline="\\n".
    - Atomic: writes to a temp file, fsyncs, then renames.
    - No logging side effects.
    """
    target = Path(path)
    data = str(text).replace("\r\n", "\n").replace("\r", "\n")
    _atomic_write(target, data)


def write_yaml_file(path: str | Path, data: Any) -> None:
    """
    Write YAML content to a file atomically with consistent UTF-8 encoding and LF newlines.

    Guarantees:
    - Converts ``path`` to ``Path``.
    - Ensures parent directories exist.
    - Uses yaml.dump with:
        - default_flow_style=False
        - sort_keys=False
        - allow_unicode=True
      (callers can rely on globally-registered representers such as _LiteralString)
    - Writes with encoding="utf-8" and newline="\\n".
    - Atomic: writes to a temp file, fsyncs, then renames.
    - No logging or global YAML configuration changes.
    """
    target = Path(path)
    yaml_text = yaml.dump(
        data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    yaml_text = yaml_text.replace("\r\n", "\n").replace("\r", "\n")
    _atomic_write(target, yaml_text)
