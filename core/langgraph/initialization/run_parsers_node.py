# core/langgraph/initialization/run_parsers_node.py
"""Run parsers to create full graph structure during initialization.

This module defines the initialization node that runs all parsers to transform
initialization artifacts into the complete knowledge graph structure.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from core.langgraph.content_manager import require_project_dir
from core.langgraph.state import NarrativeState
from core.parser_runner import ParserRunner

logger = structlog.get_logger(__name__)


async def run_initialization_parsers(state: NarrativeState) -> NarrativeState:
    """Run all parsers to create full graph structure from initialization artifacts.

    This node runs after initialization files are persisted and transforms them into
    the complete knowledge graph structure with proper nodes and relationships.

    Args:
        state: Workflow state containing project_dir.

    Returns:
        Updated state with parser results and initialization progress.

    Notes:
        This node runs all parsers in sequence:
        1. CharacterSheetParser - Creates Character nodes with relationships
        2. GlobalOutlineParser - Parses global outline structure
        3. ActOutlineParser - Creates Events and HAPPENS_BEFORE relationships
        4. ChapterOutlineParser - Creates Chapter structure

        On errors, returns a state with `has_fatal_error` set and `last_error` populated.
    """
    logger.info("run_initialization_parsers: starting parser execution")

    project_dir = require_project_dir(state)
    project_path = Path(project_dir)

    try:
        runner = ParserRunner(project_path)
        results = await runner.run_all_parsers()

        failed_parsers = [name for name, (success, _) in results.items() if not success]

        if failed_parsers:
            error_messages = [f"{name}: {msg}" for name, (success, msg) in results.items() if not success]
            error_msg = f"Parser failures: {', '.join(error_messages)}"
            logger.error(
                "run_initialization_parsers: parser failures detected",
                failed_parsers=failed_parsers,
                error=error_msg,
            )
            return {
                **state,
                "current_node": "run_parsers",
                "last_error": error_msg,
                "has_fatal_error": True,
                "error_node": "run_parsers",
                "initialization_step": "parsers_failed",
            }

        logger.info(
            "run_initialization_parsers: all parsers completed successfully",
            parsers=list(results.keys()),
        )

        return {
            **state,
            "current_node": "run_parsers",
            "last_error": None,
            "initialization_step": "parsers_complete",
        }

    except Exception as e:
        error_msg = f"Failed to run parsers: {e}"
        logger.error(
            "run_initialization_parsers: fatal error during parser execution",
            error=str(e),
            exc_info=True,
        )
        return {
            **state,
            "current_node": "run_parsers",
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "run_parsers",
            "initialization_step": "parsers_failed",
        }


__all__ = ["run_initialization_parsers"]
