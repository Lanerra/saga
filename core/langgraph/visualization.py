# core/langgraph/visualization.py
"""Render LangGraph workflows for debugging and inspection.

This module can export a compiled workflow graph to:
- Mermaid diagrams (`.md` / `.mmd`).
- PNG images (requires Graphviz integration).
- Plain-text ASCII summaries.

Notes:
    PNG export depends on optional Graphviz tooling via LangChain/LangGraph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import structlog

logger = structlog.get_logger(__name__)


def visualize_workflow(
    graph: Any,
    output_path: str | Path,
    format: Literal["mermaid", "png", "ascii"] = "mermaid",
    title: str | None = None,
) -> Path:
    """Write a visualization of a compiled LangGraph workflow.

    Args:
        graph: Compiled workflow graph.
        output_path: Destination file path.
        format: Export format.
        title: Optional diagram title.

    Returns:
        Path to the written file.

    Raises:
        ImportError: When `format="png"` and PNG export dependencies are unavailable.
        ValueError: When `format` is not supported.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "visualize_workflow: generating visualization",
        format=format,
        output_path=str(output_path),
    )

    if format == "mermaid":
        _export_mermaid(graph, output_path, title)
    elif format == "png":
        _export_png(graph, output_path, title)
    elif format == "ascii":
        _export_ascii(graph, output_path, title)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(
        "visualize_workflow: visualization created",
        path=str(output_path),
        size_bytes=output_path.stat().st_size,
    )

    return output_path


def _export_mermaid(graph: Any, output_path: Path, title: str | None) -> None:
    """Export the workflow as Mermaid source."""
    try:
        # LangGraph's built-in Mermaid export
        mermaid_code = graph.get_graph().draw_mermaid()

        # Add title if provided
        if title:
            mermaid_with_title = f"---\ntitle: {title}\n---\n{mermaid_code}"
        else:
            mermaid_with_title = mermaid_code

        # Write to file
        output_path.write_text(mermaid_with_title, encoding="utf-8")

        logger.debug(
            "_export_mermaid: Mermaid diagram created",
            path=str(output_path),
        )

    except Exception as e:
        logger.error(
            "_export_mermaid: failed to create Mermaid diagram",
            error=str(e),
            exc_info=True,
        )
        raise


def _export_png(graph: Any, output_path: Path, title: str | None) -> None:
    """Export the workflow as a PNG image.

    Raises:
        ImportError: When Graphviz integration is unavailable.
    """
    try:
        # Try to use LangGraph's built-in PNG export via graphviz
        from langchain_core.runnables.graph import CurveStyle

        png_data = graph.get_graph().draw_mermaid_png(
            curve_style=CurveStyle.LINEAR,
        )

        # Write PNG data to file
        output_path.write_bytes(png_data)

        logger.debug(
            "_export_png: PNG diagram created",
            path=str(output_path),
        )

    except ImportError as e:
        logger.error(
            "_export_png: graphviz not available",
            error=str(e),
        )
        raise ImportError("PNG export requires graphviz. Install with: pip install pygraphviz") from e
    except Exception as e:
        logger.error(
            "_export_png: failed to create PNG diagram",
            error=str(e),
            exc_info=True,
        )
        raise


def _export_ascii(graph: Any, output_path: Path, title: str | None) -> None:
    """Export the workflow as a plain-text summary."""
    try:
        # Get graph structure
        graph_obj = graph.get_graph()

        # Build ASCII representation
        lines = []

        if title:
            lines.append(f"# {title}")
            lines.append("=" * len(f"# {title}"))
            lines.append("")

        lines.append("Nodes:")
        lines.append("-" * 40)
        for node_id in graph_obj.nodes:
            lines.append(f"  • {node_id}")

        lines.append("")
        lines.append("Edges:")
        lines.append("-" * 40)

        for edge in graph_obj.edges:
            source = edge.source
            target = edge.target
            if edge.conditional:
                lines.append(f"  {source} --[conditional]--> {target}")
            else:
                lines.append(f"  {source} --> {target}")

        lines.append("")
        lines.append(f"Entry Point: {graph_obj.entry_point if hasattr(graph_obj, 'entry_point') else 'unknown'}")
        lines.append("")

        ascii_diagram = "\n".join(lines)
        output_path.write_text(ascii_diagram, encoding="utf-8")

        logger.debug(
            "_export_ascii: ASCII diagram created",
            path=str(output_path),
        )

    except Exception as e:
        logger.error(
            "_export_ascii: failed to create ASCII diagram",
            error=str(e),
            exc_info=True,
        )
        raise


def print_workflow_summary(graph: Any, title: str | None = None) -> None:
    """Print a workflow summary to stdout."""
    try:
        graph_obj = graph.get_graph()

        if title:
            print(f"\n{'=' * 60}")
            print(f"  {title}")
            print(f"{'=' * 60}\n")

        print(f"Nodes ({len(graph_obj.nodes)}):")
        for i, node_id in enumerate(graph_obj.nodes, 1):
            print(f"  {i}. {node_id}")

        print(f"\nEdges ({len(graph_obj.edges)}):")
        for i, edge in enumerate(graph_obj.edges, 1):
            source = edge.source
            target = edge.target
            print(f"  {i}. {source} → {target}")

        print()

    except Exception as e:
        logger.error(
            "print_workflow_summary: failed to print summary",
            error=str(e),
            exc_info=True,
        )


__all__ = [
    "visualize_workflow",
    "print_workflow_summary",
]
