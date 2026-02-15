"""Tests for workflow visualization export functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest

from core.langgraph.visualization import print_workflow_summary, visualize_workflow


@dataclass
class FakeEdge:
    source: str
    target: str
    conditional: bool = False


@dataclass
class FakeGraphObject:
    nodes: dict[str, str] = field(default_factory=dict)
    edges: list[FakeEdge] = field(default_factory=list)
    entry_point: str = "start"

    def draw_mermaid(self) -> str:
        return "graph TD\n  A --> B"

    def draw_mermaid_png(self, **kwargs: object) -> bytes:
        return b"fake-png-data"


class FakeGraph:
    def __init__(self, graph_object: FakeGraphObject) -> None:
        self._graph_object = graph_object

    def get_graph(self) -> FakeGraphObject:
        return self._graph_object


def _build_fake_graph() -> FakeGraph:
    graph_object = FakeGraphObject(
        nodes={"start": "start", "process": "process", "end": "end"},
        edges=[
            FakeEdge(source="start", target="process"),
            FakeEdge(source="process", target="end"),
        ],
        entry_point="start",
    )
    return FakeGraph(graph_object)


class TestVisualizeMermaid:
    """Verify mermaid export produces correct file content."""

    def test_creates_file_with_mermaid_content(self, tmp_path: Path) -> None:
        """Exported file contains the mermaid diagram source."""
        graph = _build_fake_graph()
        output = tmp_path / "diagram.mmd"

        result = visualize_workflow(graph, output, format="mermaid")

        assert result == output
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert content == "graph TD\n  A --> B"

    def test_with_title_prepends_yaml_frontmatter(self, tmp_path: Path) -> None:
        """When a title is provided, YAML frontmatter is prepended."""
        graph = _build_fake_graph()
        output = tmp_path / "titled.mmd"

        visualize_workflow(graph, output, format="mermaid", title="My Workflow")

        content = output.read_text(encoding="utf-8")
        assert content == "---\ntitle: My Workflow\n---\ngraph TD\n  A --> B"

    def test_without_title_has_no_frontmatter(self, tmp_path: Path) -> None:
        """Without a title, the file starts directly with the mermaid source."""
        graph = _build_fake_graph()
        output = tmp_path / "no_title.mmd"

        visualize_workflow(graph, output, format="mermaid")

        content = output.read_text(encoding="utf-8")
        assert not content.startswith("---")
        assert content == "graph TD\n  A --> B"


class TestVisualizeAscii:
    """Verify ASCII export produces correct plain-text output."""

    def test_creates_file_with_nodes_and_edges(self, tmp_path: Path) -> None:
        """Exported file contains nodes section and edges section."""
        graph = _build_fake_graph()
        output = tmp_path / "diagram.txt"

        result = visualize_workflow(graph, output, format="ascii")

        assert result == output
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "Nodes:" in content
        assert "Edges:" in content

    def test_with_title_includes_title_header(self, tmp_path: Path) -> None:
        """When a title is provided, it appears as a header at the top."""
        graph = _build_fake_graph()
        output = tmp_path / "titled.txt"

        visualize_workflow(graph, output, format="ascii", title="Pipeline Overview")

        content = output.read_text(encoding="utf-8")
        lines = content.split("\n")
        assert lines[0] == "# Pipeline Overview"
        assert lines[1] == "=" * len("# Pipeline Overview")

    def test_lists_all_nodes_and_edges(self, tmp_path: Path) -> None:
        """All node names and edge connections appear in the output."""
        graph = _build_fake_graph()
        output = tmp_path / "full.txt"

        visualize_workflow(graph, output, format="ascii")

        content = output.read_text(encoding="utf-8")
        for node_name in ("start", "process", "end"):
            assert f"  \u2022 {node_name}" in content
        assert "  start --> process" in content
        assert "  process --> end" in content
        assert "Entry Point: start" in content


class TestVisualizePng:
    """Verify PNG export writes binary data to file."""

    def test_writes_png_bytes_to_file(self, tmp_path: Path) -> None:
        """PNG export writes the bytes returned by draw_mermaid_png."""
        graph = _build_fake_graph()
        output = tmp_path / "diagram.png"

        with patch(
            "core.langgraph.visualization.CurveStyle",
            create=True,
        ):
            from unittest.mock import MagicMock

            fake_curve_style = MagicMock()
            fake_curve_style.LINEAR = "linear"
            with patch.dict(
                "sys.modules",
                {"langchain_core.runnables.graph": MagicMock(CurveStyle=fake_curve_style)},
            ):
                result = visualize_workflow(graph, output, format="png")

        assert result == output
        assert output.exists()
        assert output.read_bytes() == b"fake-png-data"


class TestVisualizeErrors:
    """Verify error handling for unsupported formats and directory creation."""

    def test_unsupported_format_raises_value_error(self, tmp_path: Path) -> None:
        """An unrecognized format string raises ValueError."""
        graph = _build_fake_graph()
        output = tmp_path / "diagram.svg"

        with pytest.raises(ValueError, match="Unsupported format: svg"):
            visualize_workflow(graph, output, format="svg")  # type: ignore[arg-type]

    def test_parent_directories_created_automatically(self, tmp_path: Path) -> None:
        """Nested output paths have their parent directories created."""
        graph = _build_fake_graph()
        output = tmp_path / "deep" / "nested" / "dir" / "diagram.mmd"

        assert not output.parent.exists()

        result = visualize_workflow(graph, output, format="mermaid")

        assert result == output
        assert output.exists()


class TestPrintWorkflowSummary:
    """Verify print_workflow_summary stdout output."""

    def test_prints_node_count_and_edge_details(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Output includes the node count and numbered edge list."""
        graph = _build_fake_graph()

        print_workflow_summary(graph)

        captured = capsys.readouterr().out
        assert "Nodes (3):" in captured
        assert "Edges (2):" in captured
        assert "1. start \u2192 process" in captured
        assert "2. process \u2192 end" in captured

    def test_with_title_prints_title_header(self, capsys: pytest.CaptureFixture[str]) -> None:
        """When a title is given, it appears in a decorated header."""
        graph = _build_fake_graph()

        print_workflow_summary(graph, title="Generation Workflow")

        captured = capsys.readouterr().out
        assert "Generation Workflow" in captured
        assert "=" * 60 in captured
