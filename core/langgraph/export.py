# core/langgraph/export.py
"""Export project chapter files into consolidated artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from utils.file_io import write_text_file


def _iter_chapter_files(chapters_dir: Path) -> Iterable[Path]:
    """Yield canonical chapter Markdown files sorted by chapter number.

    Notes:
        Canonical chapters are files matching `chapter_<NNN>.md` in `chapters_dir`. Only
        `.md` files are considered canonical for export (no `.txt` mirrors).
    """
    if not chapters_dir.is_dir():
        return []

    # Collect candidates first (chapter_*.md), then sort by parsed chapter number.
    chapter_files: list[tuple[int, Path]] = []
    for path in chapters_dir.glob("chapter_*.md"):
        stem = path.stem  # e.g. "chapter_001"
        # Defensive parse: expect "chapter_" prefix followed by an int.
        if not stem.startswith("chapter_"):
            continue
        num_str = stem[len("chapter_") :]
        if not num_str.isdigit():
            continue
        chapter_num = int(num_str)
        chapter_files.append((chapter_num, path))

    chapter_files.sort(key=lambda item: item[0])
    return [p for _, p in chapter_files]


def _strip_bom(text: str) -> str:
    """Remove a UTF-8 BOM if present."""
    if text.startswith("\ufeff"):
        return text[1:]
    return text


def _extract_body(content: str) -> str:
    """Extract chapter prose, removing optional YAML front matter.

    Notes:
        This strips YAML front matter only when the file begins with a line that is
        exactly `---` and a matching closing `---` delimiter appears later.
        If the front matter is malformed (missing a closing delimiter), the function
        treats the full file as the body.

        The body is normalized to real newlines and only leading newlines at the very
        start are removed to preserve internal formatting.
    """
    content = _strip_bom(content)

    # Normalize Windows line endings defensively.
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    if content.startswith("---"):
        # Split into lines for predictable scanning.
        lines = content.split("\n")
        # First line is '---'; search for closing '---'.
        end_index = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_index = i
                break

        if end_index is not None:
            # Body starts after the closing '---' line.
            body_lines = lines[end_index + 1 :]
            body = "\n".join(body_lines)
        else:
            # Malformed front matter (no closing '---'); fail gracefully:
            # treat entire content as body.
            body = content
    else:
        body = content

    # Replace any literal backslash-n sequences with real newlines just in case,
    # but this is mainly defensive; upstream writers already emit real newlines.
    body = body.replace("\\n", "\n")

    # Trim only leading newlines at very start; preserve internal spacing.
    body = body.lstrip("\n")

    return body


def generate_full_export(project_dir: str | Path) -> Path:
    """Generate a full novel export by concatenating chapter Markdown files.

    Args:
        project_dir: Project root directory containing `chapters/` and `exports/`.

    Returns:
        Path to the intended export file (`exports/novel_full.md`). If no canonical
        chapter files exist, the function returns this path without creating it.

    Notes:
        - Chapters are discovered from `chapters/chapter_<NNN>.md` files and sorted by
          the numeric component.
        - YAML front matter is removed when present and well-formed.
        - Chapter bodies are separated by exactly one blank line in the final output.
    """
    project_path = Path(project_dir)
    chapters_dir = project_path / "chapters"
    exports_dir = project_path / "exports"
    output_path = exports_dir / "novel_full.md"

    chapter_files = list(_iter_chapter_files(chapters_dir))
    if not chapter_files:
        # No canonical chapters: do nothing; return the planned path so callers
        # have a stable reference without forcing an empty artifact.
        return output_path

    bodies: list[str] = []
    for chapter_path in chapter_files:
        # Explicit UTF-8 read; errors='replace' to avoid crashing on rare issues,
        # but production files should always be well-formed UTF-8.
        raw = chapter_path.read_text(encoding="utf-8", errors="replace")
        body = _extract_body(raw)
        # Skip if body is empty after normalization; we still include a placeholder
        # empty segment via rstrip/joining rules, which is acceptable
        bodies.append(body)

    # Prepare final payload:
    # - Strip trailing spaces/newlines per body.
    # - Join with exactly one blank line between chapters.
    normalized_segments = [b.rstrip() for b in bodies]
    final_text = "\n\n".join(normalized_segments)
    # Ensure file ends with a newline for POSIX-friendly semantics.
    if not final_text.endswith("\n"):
        final_text += "\n"

    write_text_file(output_path, final_text)

    return output_path
