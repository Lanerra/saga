from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

import core.langgraph.initialization.persist_files_node as persistence
import core.parser_runner as parser_runner
from core.langgraph.content_manager import ContentManager
from core.langgraph.state import NarrativeState
from core.langgraph.workflow import create_full_workflow_graph, should_initialize


def _state(project: Path, names: tuple[str, ...] = ("Hero",)) -> NarrativeState:
    manager = ContentManager(str(project))
    return {
        "project_dir": str(project),
        "character_sheets_ref": manager.save_json({name: {"description": "Synthetic biography"} for name in names}, "character_sheets", "all", 1),
        "global_outline_ref": manager.save_json({"raw_text": "Synthetic outline"}, "global_outline", "main", 1),
        "act_outlines_ref": manager.save_json({"1": {"raw_text": "Synthetic act"}}, "act_outlines", "all", 1),
    }


def _snapshot(project: Path) -> dict[str, str]:
    return {
        str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "directory"
        for path in project.rglob("*")
    }


@pytest.mark.parametrize("relative", [
    "world/rules.yaml", "world/history.yaml", "world/items.yaml", "characters/hero.yaml",
    "characters/obsolete.yaml", "characters/HERO.YAML", "World/Rules.yaml", "saga.yaml",
    "outline/beats.yaml", "outline/structure.yaml", "summaries/README.md", "chapters/chapter_001.md",
])
@pytest.mark.parametrize("content", [b"", b"# User-owned bytes\r\nitems: [keepsake]\r\n"])
async def test_existing_artifacts_block_all_persistence(tmp_path: Path, relative: str, content: bytes) -> None:
    state = _state(tmp_path)
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    before = _snapshot(tmp_path)

    result = await persistence.persist_initialization_files(state)

    assert _snapshot(tmp_path) == before
    assert result["has_fatal_error"] is True
    assert result["error_node"] == "persist_files"
    assert result["initialization_step"] == "file_persistence_failed"
    assert result["last_error"] is not None
    assert "Reinitialization is blocked" in result["last_error"]


@pytest.mark.parametrize("names", [
    ("A B", "A_B"), ("O'Neil", "ONeil"), ("Hero", "HERO"),
    ("Straße", "STRASSE"), ("Élan", "E\u0301lan"), ("Ａ", "A"),
])
async def test_colliding_names_fail_before_any_projection(tmp_path: Path, names: tuple[str, ...]) -> None:
    state = _state(tmp_path, names)
    before = _snapshot(tmp_path)

    result = await persistence.persist_initialization_files(state)

    assert _snapshot(tmp_path) == before
    assert result["has_fatal_error"] is True
    assert result["last_error"] is not None
    assert "Character projection collision" in result["last_error"]


@pytest.mark.parametrize("name", ["../escape", "A/B", "A\\B", "NUL", "con.txt", "name.", "A:B", "", "'", "A\x00B"])
async def test_nonportable_names_fail_before_any_projection(tmp_path: Path, name: str) -> None:
    state = _state(tmp_path, (name,))
    before = _snapshot(tmp_path)

    result = await persistence.persist_initialization_files(state)

    assert _snapshot(tmp_path) == before
    assert result["has_fatal_error"] is True
    assert result["last_error"] is not None
    assert "Invalid character projection name" in result["last_error"]


async def test_reinitialization_preserves_first_publication(tmp_path: Path) -> None:
    state = _state(tmp_path)
    first = await persistence.persist_initialization_files(state)
    assert first["initialization_step"] == "files_persisted"
    before = _snapshot(tmp_path)

    second = await persistence.persist_initialization_files(state)

    assert _snapshot(tmp_path) == before
    assert second["has_fatal_error"] is True
    assert second["last_error"] is not None
    assert "Reinitialization is blocked" in second["last_error"]


async def test_partial_publication_is_fatal_and_retry_preserves_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(tmp_path)
    original_link = os.link

    def fail_late(source: Path, destination: Path) -> None:
        if destination.name == "items.yaml":
            raise OSError("Synthetic publication failure")
        original_link(source, destination)

    with monkeypatch.context() as failure:
        failure.setattr(os, "link", fail_late)
        result = await persistence.persist_initialization_files(state)

    assert result["has_fatal_error"] is True
    assert result["error_node"] == "persist_files"
    assert result["last_error"] is not None
    assert "Synthetic publication failure" in result["last_error"]
    assert (tmp_path / "characters/hero.yaml").is_file()
    assert (tmp_path / "world/items.yaml").exists() is False
    before = _snapshot(tmp_path)
    retry = await persistence.persist_initialization_files(state)
    assert retry["has_fatal_error"] is True
    assert _snapshot(tmp_path) == before
    assert sorted(path.name for path in (tmp_path / "world").iterdir()) == []


async def test_destination_created_during_publication_is_not_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(tmp_path)
    original_link = os.link
    user_content = b"name: User replacement\n"

    def create_collision(source: Path, destination: Path) -> None:
        if destination.name == "hero.yaml":
            destination.write_bytes(user_content)
        original_link(source, destination)

    monkeypatch.setattr(os, "link", create_collision)
    result = await persistence.persist_initialization_files(state)

    assert result["has_fatal_error"] is True
    assert (tmp_path / "characters/hero.yaml").read_bytes() == user_content
    assert sorted(path.name for path in (tmp_path / "characters").iterdir()) == ["hero.yaml"]
    assert (tmp_path / "saga.yaml").exists() is False


@pytest.mark.parametrize("relative", ["saga.yaml", "world/items.yaml", "characters/obsolete.yaml", ".saga/content/character_sheets/all_v1.json"])
def test_workflow_admission_blocks_existing_project(tmp_path: Path, relative: str) -> None:
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"synthetic prior artifact")
    before = _snapshot(tmp_path)

    with pytest.raises(FileExistsError, match="Reinitialization is blocked"):
        should_initialize({"project_dir": str(tmp_path), "initialization_complete": False})

    assert _snapshot(tmp_path) == before
    assert should_initialize({"project_dir": str(tmp_path), "initialization_complete": True}) == "generate"


def test_new_workflow_admission_is_read_only(tmp_path: Path) -> None:
    assert should_initialize({"project_dir": str(tmp_path), "initialization_complete": False}) == "initialize"
    assert _snapshot(tmp_path) == {}


@pytest.mark.parametrize("versions", [(0, 1), (1,)])
@pytest.mark.parametrize("selection", ["all", "character_sheets", "global_outline", "act_outlines", "chapter_outlines"])
async def test_active_plan_blocks_every_legacy_parser_before_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, versions: tuple[int, ...], selection: str,
) -> None:
    manager = ContentManager(str(tmp_path))
    for version in versions:
        manager.save_json({"1": {"raw_text": f"Synthetic version {version}"}}, "chapter_outlines", "all", version)
    runner = parser_runner.ParserRunner(tmp_path)
    before = _snapshot(tmp_path)
    constructed: list[str] = []

    def unexpected_construction(self: Any, **arguments: Any) -> None:
        constructed.append(type(self).__name__)
        raise AssertionError("A parser must not start for an active plan")

    for parser_class in [parser_runner.CharacterSheetParser, parser_runner.GlobalOutlineParser, parser_runner.ActOutlineParser, parser_runner.ChapterOutlineParser]:
        monkeypatch.setattr(parser_class, "__init__", unexpected_construction)

    if selection == "all":
        results = await runner.run_all_parsers()
    else:
        results = {selection: await runner.run_parser(selection)}

    assert constructed == []
    assert _snapshot(tmp_path) == before
    assert results == ({"initialization": (False, "Initialization import failed: No frozen initialization selected; legacy filename replay is blocked")} if selection == "all" else {
        selection: (False, "Individual parser writes are blocked; prepare and accept one complete initialization import")
    })


def test_direct_parser_construction_rejects_active_plan(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))
    manager.save_json({"1": {"raw_text": "Synthetic active plan"}}, "chapter_outlines", "all", 1)
    runner = parser_runner.ParserRunner(tmp_path)
    before = _snapshot(tmp_path)
    with pytest.raises(FileExistsError, match="Parser replay is blocked"):
        runner._create_parser_instance(parser_runner.ChapterOutlineParser)
    assert _snapshot(tmp_path) == before


def test_initial_skeleton_requires_explicit_import(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))
    reference = manager.save_json({"1": {"raw_text": "Synthetic initial plan"}}, "chapter_outlines", "all", 0)
    runner = parser_runner.ParserRunner(tmp_path)
    before = (tmp_path / reference["path"]).read_bytes()
    with pytest.raises(FileExistsError, match="Parser replay is blocked"):
        runner._create_parser_instance(parser_runner.ChapterOutlineParser)
    assert (tmp_path / reference["path"]).read_bytes() == before


def test_changed_module_imports_resolve_to_this_tree() -> None:
    root = Path(__file__).resolve().parents[1]
    assert Path(persistence.__file__).resolve() == root / "core/langgraph/initialization/persist_files_node.py"
    assert Path(parser_runner.__file__).resolve() == root / "core/parser_runner.py"


@pytest.mark.parametrize("relative", ["world/rules.yaml", ".saga/content/character_sheets/all_v1.json"])
async def test_compiled_workflow_rejects_reinitialization_before_generation(tmp_path: Path, relative: str) -> None:
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"Synthetic prior project bytes")
    before = _snapshot(tmp_path)
    workflow = create_full_workflow_graph()

    with pytest.raises(FileExistsError, match="Reinitialization is blocked"):
        await workflow.ainvoke({"project_dir": str(tmp_path), "initialization_complete": False})

    assert _snapshot(tmp_path) == before


async def test_existing_user_file_is_checked_before_content_manager_creation(tmp_path: Path) -> None:
    world = tmp_path / "world"
    world.mkdir()
    (world / "history.yaml").write_bytes(b"events: [Synthetic user history]\n")
    before = _snapshot(tmp_path)

    result = await persistence.persist_initialization_files({"project_dir": str(tmp_path)})

    assert result["has_fatal_error"] is True
    assert _snapshot(tmp_path) == before
    assert (tmp_path / ".saga").exists() is False


@pytest.mark.parametrize("relative", ["world", "characters", "outline", "summaries", "chapters", "exports"])
async def test_linked_projection_directories_are_not_written(tmp_path: Path, relative: str) -> None:
    project = tmp_path / "project"
    state = _state(project)
    external = tmp_path / "synthetic-external"
    external.mkdir()
    (project / relative).symlink_to(external, target_is_directory=True)
    before = _snapshot(project)

    result = await persistence.persist_initialization_files(state)

    assert result["has_fatal_error"] is True
    assert _snapshot(project) == before
    assert _snapshot(external) == {}


@pytest.mark.parametrize("relative", ["world/rules.yaml", "characters/hero.yaml", "saga.yaml"])
async def test_dangling_user_projection_links_are_preserved(tmp_path: Path, relative: str) -> None:
    state = _state(tmp_path)
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(tmp_path / "absent-user-file")
    before = _snapshot(tmp_path)

    result = await persistence.persist_initialization_files(state)

    assert result["has_fatal_error"] is True
    assert _snapshot(tmp_path) == before
    assert target.is_symlink() is True
    assert target.readlink() == tmp_path / "absent-user-file"


@pytest.mark.parametrize("kind", ["empty", "directory", "dangling-link"])
async def test_unreadable_active_plan_is_not_permission_to_replay(tmp_path: Path, kind: str) -> None:
    runner = parser_runner.ParserRunner(tmp_path)
    plan = tmp_path / ".saga/content/chapter_outlines/all_v1.json"
    plan.parent.mkdir(parents=True)
    if kind == "empty":
        plan.write_bytes(b"")
    elif kind == "directory":
        plan.mkdir()
    else:
        plan.symlink_to(tmp_path / "absent-plan")
    before = _snapshot(tmp_path)

    result = await runner.run_parser("chapter_outlines")

    assert result == (
        False,
        "Individual parser writes are blocked; prepare and accept one complete initialization import",
    )
    assert _snapshot(tmp_path) == before
