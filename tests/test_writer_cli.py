"""Writer boundaries use disposable projects and real file/CLI operations."""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from typing import Any

import pytest

import main
from core.langgraph.export import _extract_body, generate_full_export, generate_legacy_export
from core.langgraph.manuscript import ManuscriptStore
from core.project_config import NarrativeProjectConfig
from core.project_manager import ProjectManager


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> NarrativeProjectConfig:
    monkeypatch.setattr(ProjectManager, "projects_root", tmp_path / "projects")
    return NarrativeProjectConfig(title="Synthetic Story", genre="Mystery", theme="Discovery", setting="Archive", protagonist_name="Ada", narrative_style="First person", total_chapters=2)


@pytest.mark.parametrize("original_review", [False, True])
@pytest.mark.parametrize("new_review", [False, True])
@pytest.mark.parametrize("title", ["Synthetic Story", "Synthetic-Story!"])
def test_collision_preserves_every_existing_byte(project: NarrativeProjectConfig, original_review: bool, new_review: bool, title: str) -> None:
    directory = ProjectManager.save_config(project, review=original_review)
    marker = directory / "author.txt"
    marker.write_bytes(b"Author bytes.\r\n")
    before = {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    changed = project.model_copy(update={"title": title, "theme": "Different"})
    with pytest.raises(FileExistsError):
        ProjectManager.save_config(changed, review=new_review)
    assert {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()} == before


def test_config_publication_failure_leaves_no_partial_config(project: NarrativeProjectConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_link(*arguments: Any, **keywords: Any) -> None:
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="synthetic publication failure"):
        ProjectManager.save_config(project, review=False)
    directory = ProjectManager.projects_root / "synthetic_story"
    assert not (directory / "config.json").exists()


def test_invalid_candidate_is_not_promoted(project: NarrativeProjectConfig) -> None:
    directory = ProjectManager.save_config(project, review=True)
    candidate = directory / "config.candidate.json"
    candidate.write_bytes(b'{"title": "incomplete"}')
    with pytest.raises(ValueError):
        ProjectManager.promote_candidate(directory)
    assert candidate.read_bytes() == b'{"title": "incomplete"}'
    assert not (directory / "config.json").exists()


def test_promotion_race_preserves_winner(project: NarrativeProjectConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = ProjectManager.save_config(project, review=True)
    original_link = os.link
    candidate = (directory / "config.candidate.json").read_bytes()

    def competing_link(source: Any, destination: Any, **keywords: Any) -> None:
        (directory / "config.json").write_bytes(b"existing author config")
        original_link(source, destination, **keywords)

    monkeypatch.setattr(os, "link", competing_link)
    with pytest.raises(FileExistsError):
        ProjectManager.promote_candidate(directory)
    assert (directory / "config.json").read_bytes() == b"existing author config"
    assert (directory / "config.candidate.json").read_bytes() == candidate


@pytest.mark.parametrize("body", ["---A dialogue dash\nwords\n---\nending", "\\nLiteral escapes.  \r\n", "\n\nLeading prose\n\n", "\ufeffBody BOM", "---\nnot closed", "--- \nnot frontmatter\n---\nending", "---\nmetadata\n ---\nnot closed"])
def test_body_without_exact_frontmatter_is_unchanged(body: str) -> None:
    assert _extract_body(body) == body


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_exact_frontmatter_preserves_body(newline: str) -> None:
    body = "\n\\nDialogue.  \r\n\n"
    assert _extract_body(f"---{newline}chapter: 1{newline}---{newline}" + body) == body


def test_legacy_export_preserves_prose(tmp_path: Path) -> None:
    (tmp_path / "chapters").mkdir()
    body = "\n---A dialogue dash\n---\n\\nProse.  \r\n"
    (tmp_path / "chapters/chapter_001.md").write_bytes(body.encode())
    assert generate_legacy_export(tmp_path).read_bytes() == body.encode()


@pytest.mark.parametrize("names", [[], ["chapter_002.md"], ["chapter_001.md", "chapter_003.md"], ["chapter_1.md", "chapter_001.md"]])
def test_legacy_export_rejects_empty_gapped_duplicate_input(tmp_path: Path, names: list[str]) -> None:
    (tmp_path / "chapters").mkdir()
    for name in names:
        (tmp_path / "chapters" / name).write_text("Synthetic body")
    with pytest.raises(ValueError):
        generate_legacy_export(tmp_path)
    assert not (tmp_path / "exports/novel_legacy_unverified.md").exists()


@pytest.mark.parametrize("numbers", [[], [2], [1, 3], [1]])
def test_complete_export_requires_exact_chapters(tmp_path: Path, numbers: list[int]) -> None:
    store = ManuscriptStore(tmp_path)
    for number in numbers:
        store.accept(store.prepare(number, f"Synthetic {number}"))
    with pytest.raises(ValueError, match="Expected accepted chapters"):
        generate_full_export(tmp_path, expected_chapters=2)
    assert not (tmp_path / "exports/novel_full.md").exists()


def test_completion_ignores_all_unaccepted_markdown(project: NarrativeProjectConfig) -> None:
    directory = ProjectManager.save_config(project, review=False)
    (directory / "chapters").mkdir()
    for name in ["chapter_001.md", "chapter_1.md", "chapter_notes.md", "chapter_003.md"]:
        (directory / "chapters" / name).write_text("Draft")
    assert ProjectManager.count_completed_chapters(directory) == 0


def test_ambiguous_resume_is_rejected(project: NarrativeProjectConfig) -> None:
    ProjectManager.save_config(project, review=False)
    ProjectManager.save_config(project.model_copy(update={"title": "Other"}), review=False)
    with pytest.raises(ValueError, match="Multiple"):
        ProjectManager.find_resume_project()


@pytest.mark.parametrize("arguments", [[], ["generate"], ["parse"]])
def test_cli_requires_explicit_selection(arguments: list[str], monkeypatch: pytest.MonkeyPatch, project: NarrativeProjectConfig) -> None:
    candidate = ProjectManager.save_config(project, review=True)
    original = (candidate / "config.candidate.json").read_bytes()
    monkeypatch.setattr(sys, "argv", ["main.py", *arguments])
    with pytest.raises(SystemExit) as caught:
        main.main()
    assert caught.value.code == 2
    assert (candidate / "config.candidate.json").read_bytes() == original
    assert not (candidate / "config.json").exists()


def test_cli_complete_export_receipt(project: NarrativeProjectConfig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    directory = ProjectManager.save_config(project, review=False)
    store = ManuscriptStore(directory)
    bodies = ["\nFirst\\n.  \r\n", "---Second\n---\n\n"]
    for number, body in enumerate(bodies, 1):
        store.accept(store.prepare(number, body))
    monkeypatch.setattr(sys, "argv", ["main.py", "export", "--project-dir", str(directory)])
    main.main()
    assert (directory / "exports/novel_full.md").read_bytes() == b"\n\n".join(body.encode() for body in bodies)
    assert capsys.readouterr().out.splitlines()[-1] == f"SAGA export succeeded: chapters 1, 2 -> {directory / 'exports/novel_full.md'}"


def test_source_provenance() -> None:
    tree = Path(__file__).resolve().parents[1]
    assert Path(main.__file__).resolve() == tree / "main.py"
    assert Path(inspect.getfile(ProjectManager)).resolve() == tree / "core/project_manager.py"
    assert Path(inspect.getfile(generate_full_export)).resolve() == tree / "core/langgraph/export.py"


@pytest.mark.parametrize("arguments", [["--help"], ["generate", "--help"], ["bootstrap", "--help"], ["quick", "--help"], ["parse", "--help"], ["export", "--help"]])
def test_help_is_service_free(arguments: list[str], project: NarrativeProjectConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py", *arguments])
    with pytest.raises(SystemExit) as caught:
        main.main()
    assert caught.value.code == 0
    assert not ProjectManager.projects_root.exists()


def test_selected_candidate_needs_explicit_promotion(project: NarrativeProjectConfig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    directory = ProjectManager.save_config(project, review=True)
    original = (directory / "config.candidate.json").read_bytes()
    monkeypatch.setattr(sys, "argv", ["main.py", "generate", "--project-dir", str(directory)])
    with pytest.raises(SystemExit) as caught:
        main.main()
    assert caught.value.code == 1
    output = capsys.readouterr()
    assert [line for line in output.out.splitlines() if line.startswith(("Scope:", "Checkpoint:"))] == [
        "Scope: resume the selected project's durable workflow, or initialize only if fresh. No reset or deletion is requested.",
        f"Checkpoint: {directory / 'checkpoints/saga.db'}; graph ownership and retained artifacts are reconciled before continuation.",
    ]
    assert output.err == f"SAGA generate failed: Missing config.json in {directory}. No completion claimed; retained artifacts may include partial progress.\n"
    assert (directory / "config.candidate.json").read_bytes() == original
    assert not (directory / "config.json").exists()


@pytest.mark.parametrize("names", [["chapter_001.accepted.json", "chapter_1.accepted.json"], ["chapter_notes.accepted.json"]])
def test_complete_export_rejects_malformed_receipt_names(tmp_path: Path, names: list[str]) -> None:
    store = ManuscriptStore(tmp_path)
    receipt = store.prepare(1, "Synthetic exact body")
    store.accept(receipt)
    for name in names:
        (tmp_path / "chapters" / name).write_text(receipt.model_dump_json())
    with pytest.raises(ValueError, match="Invalid accepted manuscript filename"):
        generate_full_export(tmp_path, expected_chapters=1)
    assert not (tmp_path / "exports/novel_full.md").exists()


def test_legacy_invalid_utf8_preserves_previous_export(tmp_path: Path) -> None:
    (tmp_path / "chapters").mkdir()
    (tmp_path / "exports").mkdir()
    previous = tmp_path / "exports/novel_legacy_unverified.md"
    previous.write_bytes(b"Previous output")
    (tmp_path / "chapters/chapter_001.md").write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        generate_legacy_export(tmp_path)
    assert previous.read_bytes() == b"Previous output"
