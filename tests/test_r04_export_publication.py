"""Accepted export preserves author edits and avoids repeated publication."""

from pathlib import Path

import pytest

from core.langgraph.export import generate_full_export, generate_legacy_export
from core.langgraph.manuscript import ManuscriptStore


def accepted(directory: Path, number: int, text: str) -> None:
    store = ManuscriptStore(directory)
    receipt = store.prepare(number, text)
    store.accept(receipt)


def test_identical_export_keeps_publication_inode(tmp_path: Path) -> None:
    accepted(tmp_path, 1, "Synthetic accepted text.\r\n")
    output = generate_full_export(tmp_path, expected_chapters=1)
    before = output.stat()
    assert generate_full_export(tmp_path, expected_chapters=1) == output
    after = output.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


@pytest.mark.parametrize("previous_export", [False, True])
def test_export_preserves_author_changes(tmp_path: Path, previous_export: bool) -> None:
    accepted(tmp_path, 1, "Synthetic accepted text.")
    output = tmp_path / "exports/novel_full.md"
    if previous_export:
        generate_full_export(tmp_path, expected_chapters=1)
    else:
        output.parent.mkdir()
    output.write_bytes(b"Author's independent edited manuscript.\r\n")
    before = output.stat()
    with pytest.raises(ValueError, match="reconciliation|[Uu]nowned"):
        generate_full_export(tmp_path, expected_chapters=1)
    assert output.read_bytes() == b"Author's independent edited manuscript.\r\n"
    assert output.stat().st_ino == before.st_ino


def test_accepted_preview_can_extend_owned_export(tmp_path: Path) -> None:
    accepted(tmp_path, 1, "First.")
    output = generate_full_export(tmp_path)
    accepted(tmp_path, 2, "Second.")
    assert generate_full_export(tmp_path, expected_chapters=2).read_bytes() == b"First.\n\nSecond.\n"
    assert output.read_bytes() == b"First.\n\nSecond.\n"


def test_legacy_export_preserves_author_changes(tmp_path: Path) -> None:
    chapters = tmp_path / "chapters"
    chapters.mkdir()
    (chapters / "chapter_001.md").write_text("Historical prose.")
    output = generate_legacy_export(tmp_path)
    output.write_text("Author edit.")
    with pytest.raises(ValueError, match="reconciliation|[Uu]nowned"):
        generate_legacy_export(tmp_path)
    assert output.read_text() == "Author edit."
