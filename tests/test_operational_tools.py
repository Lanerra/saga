"""Offline operational boundaries and the maintained guide."""

import re
import sys
from pathlib import Path
from typing import NoReturn

import pytest

import reset_neo4j
import visualize_workflow as visualization_command
from core.langgraph.visualization import visualize_workflow

ROOT = Path(__file__).resolve().parents[1]


class UnexpectedConnection(BaseException):
    pass


class UnreachableDatabase:
    driver = None

    async def connect(self) -> NoReturn:
        raise UnexpectedConnection("Reset reached the database without project ownership")


@pytest.mark.parametrize("confirm", [False, True])
async def test_legacy_reset_refuses_before_prompt_or_connection(monkeypatch: pytest.MonkeyPatch, confirm: bool) -> None:
    monkeypatch.setattr(reset_neo4j, "neo4j_manager_instance", UnreachableDatabase(), raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    with pytest.raises(RuntimeError, match="^Database reset is disabled:"):
        await reset_neo4j.reset_neo4j_database_async("bolt://127.0.0.1:9", "synthetic", "unused", confirm=confirm)


class UnreachableRenderer:
    def get_graph(self) -> NoReturn:
        raise AssertionError("PNG reached an unconfigured renderer")


def test_png_refuses_before_renderer_or_directory_creation(tmp_path: Path) -> None:
    output = tmp_path / "absent" / "graph.png"
    with pytest.raises(ValueError, match="^PNG export is disabled:"):
        visualize_workflow(UnreachableRenderer(), output, format="png")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("format_name,extension", [("mermaid", "mmd"), ("ascii", "txt")])
def test_visualization_command_exports_real_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, format_name: str, extension: str) -> None:
    output = tmp_path / f"workflow.{extension}"
    monkeypatch.setattr(sys, "argv", ["visualize_workflow.py", "--workflow", "full", "--format", format_name, "--output", str(output)])
    visualization_command.main()
    content = output.read_text(encoding="utf-8")
    assert "init_character_sheets" in content
    assert "advance_chapter" in content
    assert "commit" in content


@pytest.mark.parametrize("path", ["README.md", "AGENTS.md", "CLAUDE.md", "docs/bootstrapper.md", "docs/langgraph-architecture.md", "docs/PROJECT_CONSTRAINTS.md"])
def test_current_guides_have_existing_local_links(path: str) -> None:
    document = ROOT / path
    content = document.read_text(encoding="utf-8")
    missing = []
    for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content):
        if "://" in target or target.startswith("#"):
            continue
        local_path = target.split("#", 1)[0]
        if not (document.parent / local_path).exists():
            missing.append(target)
    assert missing == []


def test_current_guides_do_not_advertise_removed_paths_or_implicit_selection() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/WORKFLOW_WALKTHROUGH.md" not in readme
    assert "docs/WORKFLOW_VISUALIZATION.md" not in readme
    bootstrap = (ROOT / "docs/bootstrapper.md").read_text(encoding="utf-8")
    assert "python main.py generate\n" not in bootstrap
    assert 'python main.py "A pirate adventure in space"' not in bootstrap
    assert "most recent" not in bootstrap
    for name in ["AGENTS.md", "CLAUDE.md"]:
        guidance = (ROOT / name).read_text(encoding="utf-8")
        assert 'Test names should not include the word "test"' not in guidance
        assert "TypeScript" not in guidance
        assert "expect(content)" not in guidance
        assert "core/knowledge_graph_service.py" not in guidance
        assert "core/langgraph/initialization/workflow.py" not in guidance


@pytest.mark.parametrize("path", ["docs/CURRENT_STATE_AUDIT.md", "docs/CODEBASE_AUDIT_REPORT.md", "docs/field-audit-20260505.md"])
def test_old_audits_are_explicitly_historical(path: str) -> None:
    first_lines = (ROOT / path).read_text(encoding="utf-8").splitlines()[:6]
    assert any("Historical" in line for line in first_lines)

