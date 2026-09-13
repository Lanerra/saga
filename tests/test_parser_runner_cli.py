"""Real maintenance entrypoint admission and interruption on disposable projects."""
from __future__ import annotations

import os
import runpy
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import NoReturn
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def preserve_database_ownership(run_service_context: None) -> Iterator[None]:
    from core.service_context import get_services

    database = get_services().database
    before = vars(database).copy()
    yield
    assert vars(database) == before


@pytest.mark.parametrize("selection", [None, "", "   ", "missing", "file", "traversal", "symlink", "symlink_parent"])
async def test_command_rejects_selection_before_logging_or_services(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: str | None) -> None:
    import core.parser_runner as runner
    from core.project_manager import ProjectManager
    from core.service_context import get_services

    project = tmp_path / "project"
    project.mkdir()
    (tmp_path / "file").write_text("synthetic")
    (tmp_path / "link").symlink_to(project, target_is_directory=True)
    (project / "child").mkdir()
    paths = {
        "missing": str(tmp_path / "missing"), "file": str(tmp_path / "file"),
        "traversal": str(project / ".." / "project"), "symlink": str(tmp_path / "link"),
        "symlink_parent": str(tmp_path / "link" / "child"),
    }
    selected = paths.get(selection, selection) if selection is not None else None
    logging = Mock()
    monkeypatch.setattr(runner, "setup_saga_logging", logging)
    monkeypatch.setattr(ProjectManager, "projects_root", tmp_path / "undiscovered")
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    with pytest.raises((ValueError, FileNotFoundError, NotADirectoryError)):
        await runner.run_parser_command(selected, None, services=get_services())
    logging.assert_not_called()
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == before


@pytest.mark.integration
@pytest.mark.parametrize("case", ["help", "absent", "empty", "missing", "file", "traversal", "symlink", "blocked", "failed_import", "interrupt"])
def test_standalone_entrypoint_outcomes(tmp_path: Path, case: str) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (tmp_path / "file").write_text("synthetic")
    (tmp_path / "link").symlink_to(project, target_is_directory=True)
    selections = {"empty": "", "missing": str(tmp_path / "missing"), "file": str(tmp_path / "file"), "traversal": str(project / ".." / "project"), "symlink": str(tmp_path / "link")}
    arguments = ["--help"] if case == "help" else [] if case == "absent" else ["--project-dir", selections.get(case, str(project))]
    if case == "blocked":
        arguments += ["--parser", "character_sheets"]
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    result = subprocess.run(
        [sys.executable, "-m", "tests.test_parser_runner_cli", case, str(tmp_path), *arguments],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=25,
    )
    print(result.stdout)
    expected = 0 if case == "help" else 2 if case == "absent" else 130 if case == "interrupt" else 1
    assert result.returncode == expected, result.stdout
    assert "UNEXPECTED_SERVICE" not in result.stdout
    if case not in {"interrupt", "failed_import"}:
        assert "Neo4jManagerSingleton initialized" not in result.stdout
        assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == before
        assert (tmp_path / "file").read_text() == "synthetic"
    if case == "help":
        assert "complete frozen initialization import" in " ".join(result.stdout.split())
        assert "Individual parser writes are blocked" in " ".join(result.stdout.split())
        assert "independently for testing" not in result.stdout
    elif case == "interrupt":
        assert "cancelled; no completion claimed" in result.stdout
    elif case == "blocked":
        assert "Individual parser writes are blocked" in result.stdout
    elif case == "failed_import":
        assert "No frozen initialization selected" in result.stdout
        assert "complete retained initialization accepted" not in result.stdout


def _entrypoint_probe() -> None:
    from tests.offline import OfflineBoundary

    case, directory, *arguments = sys.argv[1:]
    boundary = OfflineBoundary()
    boundary.install(change_directory=False)
    with pytest.MonkeyPatch.context() as patches:
        import config
        import core.llm_interface_refactored as interface
        import core.parser_runner as runner
        from core.project_manager import ProjectManager

        assert Path(runner.__file__).resolve() == ROOT / "core/parser_runner.py"
        print(f"Parser runner provenance: {runner.__file__}", flush=True)
        patches.setattr(ProjectManager, "projects_root", Path(directory) / "undiscovered")
        patches.setitem(vars(config), "SIMPLE_LOGGING_MODE", True)
        patches.setitem(vars(config), "ENABLE_RICH_PROGRESS", False)

        def service_creation(*arguments: object, **keywords: object) -> NoReturn:
            if case == "interrupt":
                raise KeyboardInterrupt
            raise AssertionError("UNEXPECTED_SERVICE")

        if case != "failed_import":
            patches.setattr(interface, "create_llm_service", service_creation)
        patches.setattr(sys, "argv", [str(ROOT / "core/parser_runner.py"), *arguments])
        try:
            runpy.run_path(str(ROOT / "core/parser_runner.py"), run_name="__main__")
        finally:
            boundary.close()


if __name__ == "__main__":
    _entrypoint_probe()
