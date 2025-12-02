# visualize_workflow.py
#!/usr/bin/env python3
"""
CLI tool to visualize LangGraph workflows.

Usage:
    # Generate Mermaid diagram for full workflow
    python visualize_workflow.py --workflow full --output docs/workflow_full.md

    # Generate PNG for phase2 workflow
    python visualize_workflow.py --workflow phase2 --output docs/workflow_phase2.png --format png

    # Print workflow summary to console
    python visualize_workflow.py --workflow full --summary
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Literal, cast

import structlog

from core.langgraph.visualization import print_workflow_summary, visualize_workflow
from core.langgraph.workflow import (
    create_full_workflow_graph,
    create_phase1_graph,
    create_phase2_graph,
)

logger = structlog.get_logger(__name__)


def main() -> None:
    """Main CLI entry point for workflow visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize LangGraph workflows for SAGA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate Mermaid diagram for full workflow
  python visualize_workflow.py --workflow full --output docs/workflow_full.md

  # Generate PNG for phase2 workflow
  python visualize_workflow.py --workflow phase2 --output docs/workflow_phase2.png --format png

  # Print workflow summary to console
  python visualize_workflow.py --workflow full --summary

  # Generate all workflows as Mermaid diagrams
  python visualize_workflow.py --all --output-dir docs/workflows
        """,
    )

    parser.add_argument(
        "--workflow",
        "-w",
        choices=["phase1", "phase2", "full"],
        help="Which workflow to visualize",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path (extension determines format if --format not specified)",
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["mermaid", "png", "ascii"],
        help="Output format (auto-detected from file extension if not specified)",
    )

    parser.add_argument(
        "--summary",
        "-s",
        action="store_true",
        help="Print workflow summary to console instead of creating file",
    )

    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Generate visualizations for all workflows",
    )

    parser.add_argument(
        "--output-dir",
        "-d",
        type=str,
        default="docs/workflows",
        help="Output directory when using --all (default: docs/workflows)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.workflow and not args.all:
        parser.error("Either --workflow or --all must be specified")

    if args.all and args.summary:
        parser.error("Cannot use --summary with --all")

    if not args.summary and not args.output and not args.all:
        parser.error("Either --output, --summary, or --all must be specified")

    try:
        if args.all:
            # Generate all workflows
            _generate_all_workflows(args.output_dir, args.format)
        elif args.summary:
            # Print summary to console
            _print_summary(args.workflow)
        else:
            # Generate single visualization
            _generate_single(args.workflow, args.output, args.format)

    except Exception as e:
        logger.error("Failed to generate visualization", error=str(e), exc_info=True)
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


def _generate_all_workflows(output_dir: str, format_override: str | None) -> None:
    """Generate visualizations for all workflows.

    Args:
        output_dir: Directory to save visualizations.
        format_override: Optional format to use (default: mermaid).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    format_to_use = format_override or "mermaid"
    extension = _get_extension(format_to_use)

    workflows = {
        "phase1": (create_phase1_graph, "Phase 1 Workflow (Extract, Commit, Validate)"),
        "phase2": (create_phase2_graph, "Phase 2 Workflow (Full Chapter Generation)"),
        "full": (
            create_full_workflow_graph,
            "Full SAGA Workflow (Initialization + Generation)",
        ),
    }

    print(f"Generating visualizations in {output_dir}/\n")

    for name, (factory, title) in workflows.items():
        output_file = output_path / f"workflow_{name}.{extension}"
        print(f"  • {name:8s} → {output_file.name}")

        graph = factory()
        fmt = cast(Literal["mermaid", "png", "ascii"], format_to_use)
        visualize_workflow(graph, output_file, format=fmt, title=title)

    print(f"\n✓ Generated {len(workflows)} workflow visualizations")


def _generate_single(workflow: str, output: str, format_override: str | None) -> None:
    """Generate a single workflow visualization.

    Args:
        workflow: Workflow type (phase1, phase2, full).
        output: Output file path.
        format_override: Optional format override.
    """
    # Create graph
    graph = _create_graph(workflow)

    # Determine format
    if format_override:
        format_to_use = format_override
    else:
        # Auto-detect from extension
        output_path = Path(output)
        ext = output_path.suffix.lstrip(".")
        if ext in ("md", "mmd"):
            format_to_use = "mermaid"
        elif ext == "png":
            format_to_use = "png"
        elif ext == "txt":
            format_to_use = "ascii"
        else:
            format_to_use = "mermaid"
            logger.warning(
                "Unknown extension, defaulting to Mermaid",
                extension=ext,
            )

    # Get title
    title = _get_title(workflow)

    # Generate visualization
    fmt = cast(Literal["mermaid", "png", "ascii"], format_to_use)
    result_path = visualize_workflow(graph, output, format=fmt, title=title)

    print(f"✓ Visualization created: {result_path}")


def _print_summary(workflow: str) -> None:
    """Print workflow summary to console.

    Args:
        workflow: Workflow type (phase1, phase2, full).
    """
    graph = _create_graph(workflow)
    title = _get_title(workflow)
    print_workflow_summary(graph, title=title)


def _create_graph(workflow: str) -> Any:
    """Create a workflow graph.

    Args:
        workflow: Workflow type (phase1, phase2, full).

    Returns:
        Compiled LangGraph StateGraph.
    """
    if workflow == "phase1":
        return create_phase1_graph()
    elif workflow == "phase2":
        return create_phase2_graph()
    elif workflow == "full":
        return create_full_workflow_graph()
    else:
        raise ValueError(f"Unknown workflow: {workflow}")


def _get_title(workflow: str) -> str:
    """Get title for a workflow.

    Args:
        workflow: Workflow type (phase1, phase2, full).

    Returns:
        Human-readable title.
    """
    titles = {
        "phase1": "Phase 1 Workflow (Extract, Commit, Validate)",
        "phase2": "Phase 2 Workflow (Full Chapter Generation)",
        "full": "Full SAGA Workflow (Initialization + Generation)",
    }
    return titles.get(workflow, f"Workflow: {workflow}")


def _get_extension(format: str) -> str:
    """Get file extension for a format.

    Args:
        format: Format type (mermaid, png, ascii).

    Returns:
        File extension (without dot).
    """
    extensions = {
        "mermaid": "md",
        "png": "png",
        "ascii": "txt",
    }
    return extensions.get(format, "md")


if __name__ == "__main__":
    main()
