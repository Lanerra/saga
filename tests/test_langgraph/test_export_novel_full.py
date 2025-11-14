from __future__ import annotations

from pathlib import Path

from core.langgraph.export import generate_full_export


def _write_chapter_md(path: Path, chapter_num: int, title: str, body: str) -> None:
    """Helper to create a canonical chapter .md file with YAML front matter."""
    content = (
        "---\n"
        f"chapter: {chapter_num}\n"
        f'title: "{title}"\n'
        "word_count: 4\n"
        'generated_at: "2025-01-01T00:00:00"\n'
        "version: 1\n"
        "---\n"
        "\n"
        f"{body}\n"
    )
    path.write_text(content, encoding="utf-8")


def test_generate_full_export_concatenates_chapters_in_order(tmp_path: Path) -> None:
    project_dir = tmp_path
    chapters_dir = project_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)

    # Create two canonical chapter markdown files.
    ch1 = chapters_dir / "chapter_001.md"
    ch2 = chapters_dir / "chapter_002.md"

    _write_chapter_md(ch1, 1, "Chapter 1", "This is chapter one.")
    _write_chapter_md(ch2, 2, "Chapter 2", "This is chapter two.")

    output_path = generate_full_export(str(project_dir))

    expected_output_path = project_dir / "exports" / "novel_full.md"
    assert output_path == expected_output_path
    assert expected_output_path.exists()

    content = expected_output_path.read_text(encoding="utf-8")

    # YAML front matter should be stripped.
    assert "---" not in content
    assert "chapter:" not in content
    assert "generated_at:" not in content
    assert "version:" not in content

    # Literal "\n" sequences should not appear.
    assert "\\n" not in content

    # Order and spacing:
    # - Chapter one body
    # - Exactly one blank line
    # - Chapter two body
    # A trailing newline at end of file is acceptable.
    expected = "This is chapter one.\n\nThis is chapter two.\n"
    assert content == expected


def test_generate_full_export_ignores_non_md_and_handles_missing(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path

    # No chapters directory / files: function must not raise.
    output_path = generate_full_export(str(project_dir))

    # By design (see core/langgraph/export.generate_full_export docstring):
    # - When no canonical chapter .md files are found, no export file is created.
    # - The function returns the intended output path so callers can check for existence.
    expected_output_path = project_dir / "exports" / "novel_full.md"
    assert output_path == expected_output_path
    assert not expected_output_path.exists()

    # Also verify that legacy .txt mirrors (if present) are ignored for canonical export.
    chapters_dir = project_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    (chapters_dir / "chapter_001.txt").write_text("Legacy text", encoding="utf-8")

    # Still should be a no-op because no .md chapters exist.
    output_path_after = generate_full_export(str(project_dir))
    assert output_path_after == expected_output_path
    assert not expected_output_path.exists()
