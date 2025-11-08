from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from utils.file_io import write_text_file


def _iter_chapter_files(chapters_dir: Path) -> Iterable[Path]:
    """
    Discover canonical chapter .md files and yield them sorted by chapter number.

    Assumptions:
    - Filenames follow the pattern: chapter_<NNN>.md (zero-padded integer).
    - Only .md files are considered canonical for export (no .txt mirrors).
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
    """Remove UTF-8 BOM if present."""
    if text.startswith("\ufeff"):
        return text.lstrip("\ufeff")
    return text


def _extract_body(content: str) -> str:
    """
    Extract the prose body from a chapter file, stripping YAML front matter if present.

    Rules (aligned with core/langgraph/nodes/finalize_node.py output format):
    - If the content begins with a line that is exactly '---', treat it as YAML front matter.
      Consume lines up to and including the next '---' line.
      Everything after that delimiter (and one following newline, if present)
      is treated as the chapter body.
    - If no such front matter header exists, treat the entire file as body.
    - Normalize:
      - Ensure we operate on real newlines (no literal '\\n' sequences).
      - Trim only unnecessary leading newlines at the very start of the body.
        Internal structure is preserved.
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
    """
    Generate the full novel export by concatenating canonical chapter .md files.

    Behavior:
    - Looks for chapter files under <project_dir>/chapters matching chapter_*.md.
    - Sorts chapters by the numeric component of the filename (ascending).
    - For each chapter:
        - Reads UTF-8 content.
        - Strips optional UTF-8 BOM.
        - Removes YAML front matter if present.
        - Normalizes to real newlines and trims only leading newlines at start.
    - Concatenates chapter bodies with exactly one blank line between chapters.
      (i.e., each body is rstrip()'d, then `+ "\\n\\n"` except after last one,
      where a trailing newline is still acceptable.)
    - Ensures <project_dir>/exports exists.
    - Writes result to <project_dir>/exports/novel_full.md (UTF-8).

    Edge cases:
    - If no chapter .md files are found, this function is a no-op:
        - It does not create exports/novel_full.md.
        - Returns the intended output path regardless.
      This keeps behavior simple and safe; callers can check .exists() if needed.
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
        # empty segment via rstrip/joining rules, which is acceptable.
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
