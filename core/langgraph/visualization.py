"""
LangGraph Workflow Visualization Utilities.

This module provides tools to visualize LangGraph workflows for debugging
and documentation purposes. Supports multiple output formats:
- Mermaid diagrams (.md, .mmd)
- PNG images (requires graphviz)
- ASCII text diagrams

Usage:
    from core.langgraph.visualization import visualize_workflow
    from core.langgraph.workflow import create_full_workflow_graph

    # Create graph
    graph = create_full_workflow_graph()

    # Visualize to Mermaid
    visualize_workflow(graph, "workflow.md", format="mermaid")

    # Visualize to PNG
    visualize_workflow(graph, "workflow.png", format="png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import structlog

logger = structlog.get_logger(__name__)


def visualize_workflow(
    graph,
    output_path: str | Path,
    format: Literal["mermaid", "png", "ascii"] = "mermaid",
    title: str | None = None,
) -> Path:
    """Visualize a LangGraph workflow and save to file.

    Generates a visual representation of the workflow graph showing:
    - All nodes (states/steps)
    - Edges (transitions)
    - Conditional routing
    - Entry and exit points

    Args:
        graph: Compiled LangGraph StateGraph instance.
        output_path: Path to save visualization file.
        format: Output format - "mermaid", "png", or "ascii".
        title: Optional title for the diagram.

    Returns:
        Path to the created visualization file.

    Raises:
        ImportError: If required dependencies are missing (e.g., graphviz for PNG).
        ValueError: If format is not supported.
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


def _export_mermaid(graph, output_path: Path, title: str | None) -> None:
    """Export workflow as Mermaid diagram.

    Args:
        graph: Compiled LangGraph StateGraph instance.
        output_path: Path to save Mermaid file.
        title: Optional diagram title.
    """
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


def _export_png(graph, output_path: Path, title: str | None) -> None:
    """Export workflow as PNG image.

    Requires graphviz to be installed.

    Args:
        graph: Compiled LangGraph StateGraph instance.
        output_path: Path to save PNG file.
        title: Optional diagram title.

    Raises:
        ImportError: If graphviz is not available.
    """
    try:
        # Try to use LangGraph's built-in PNG export via graphviz
        from langchain_core.runnables.graph import CurveStyle, NodeColors

        png_data = graph.get_graph().draw_mermaid_png(
            curve_style=CurveStyle.LINEAR,
            node_colors=NodeColors(
                start="#90EE90",  # Light green
                end="#FFB6C6",  # Light red
                default="#87CEEB",  # Sky blue
            ),
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
        raise ImportError(
            "PNG export requires graphviz. Install with: pip install pygraphviz"
        ) from e
    except Exception as e:
        logger.error(
            "_export_png: failed to create PNG diagram",
            error=str(e),
            exc_info=True,
        )
        raise


def _export_ascii(graph, output_path: Path, title: str | None) -> None:
    """Export workflow as ASCII text diagram.

    Creates a simple text representation of the workflow.

    Args:
        graph: Compiled LangGraph StateGraph instance.
        output_path: Path to save ASCII file.
        title: Optional diagram title.
    """
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
            # Check if it's a conditional edge
            if hasattr(edge, "conditional"):
                lines.append(f"  {source} --[conditional]--> {target}")
            else:
                lines.append(f"  {source} --> {target}")

        lines.append("")
        lines.append(
            f"Entry Point: {graph_obj.entry_point if hasattr(graph_obj, 'entry_point') else 'unknown'}"
        )
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


def print_workflow_summary(graph, title: str | None = None) -> None:
    """Print a summary of the workflow to console.

    Useful for quick debugging without creating files.

    Args:
        graph: Compiled LangGraph StateGraph instance.
        title: Optional title to display.
    """
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
