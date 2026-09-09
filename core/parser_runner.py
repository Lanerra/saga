#!/usr/bin/env python3
"""Explicit acceptance/recovery CLI for a retained complete initialization import."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import structlog

from core.logging_config import setup_saga_logging
from core.parsers.act_outline_parser import ActOutlineParser
from core.parsers.chapter_outline_parser import ChapterOutlineParser
from core.parsers.character_sheet_parser import CharacterSheetParser
from core.parsers.global_outline_parser import GlobalOutlineParser
from core.project_manager import ProjectManager
from core.service_context import RunServices, get_services, service_lifetime

logger = structlog.get_logger(__name__)


class ParserRunner:
    """Accept only an explicitly prepared complete import; never discover v0 inputs."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir

    async def run_all_parsers(self, initialization_id: str = "") -> dict[str, tuple[bool, str]]:
        from core.langgraph.initialization.staged_import import InitializationImport

        try:
            importer = InitializationImport(str(self.project_dir))
            if not importer.files.exists(f"{importer.root}/selected"):
                raise ValueError("No frozen initialization selected; legacy filename replay is blocked")
            plan = importer.load()
            identity = initialization_id or plan.identity
            database = get_services().database
            database.bind_project(plan.snapshot.project_id)
            await database.connect()
            await database.create_db_schema()
            await importer.accept(identity)
            return {"initialization": (True, f"Accepted initialization {identity}")}
        except Exception as error:
            return {"initialization": (False, f"Initialization import failed: {error}")}

    async def run_parser(self, parser_name: str) -> tuple[bool, str]:
        if parser_name not in {"character_sheets", "global_outline", "act_outlines", "chapter_outlines"}:
            raise ValueError(f"Unknown parser: {parser_name}")
        return False, "Individual parser writes are blocked; prepare and accept one complete initialization import"

    def _create_parser_instance(self, parser_class: type) -> Any:
        if parser_class not in {CharacterSheetParser, GlobalOutlineParser, ActOutlineParser, ChapterOutlineParser}:
            raise ValueError(f"Unsupported parser class: {parser_class}")
        raise FileExistsError("Parser replay is blocked; accept one complete frozen initialization import")


async def run_parser_command(project_dir_path: str | None, parser_name: str | None, *, services: RunServices | None = None) -> dict[str, tuple[bool, str]]:
    """Run the parser command with the given arguments.

    Args:
        project_dir_path: Path to the project directory. If None, use default.
        parser_name: Name of the parser to run. If None, run all parsers.

    Returns:
        Dictionary containing results for each parser.
    """
    setup_saga_logging()

    resolved_project_dir: Path
    if project_dir_path is None:
        found = ProjectManager.find_resume_project()
        if found is None:
            resolved_project_dir = ProjectManager.create_default_project()
        else:
            resolved_project_dir = found
    else:
        resolved_project_dir = Path(project_dir_path)

    logger.info("Running parsers", project_dir=str(resolved_project_dir), parser=parser_name)

    runner = ParserRunner(resolved_project_dir)

    async with service_lifetime(services):
        if parser_name:
            success, message = await runner.run_parser(parser_name)
            return {parser_name: (success, message)}
        else:
            return await runner.run_all_parsers()


def main() -> None:
    """Main entry point for the parser runner CLI."""
    parser = argparse.ArgumentParser(description="Run SAGA parsers independently for testing and debugging purposes")
    parser.add_argument(
        "--project-dir",
        "-p",
        type=str,
        help="Path to the project directory containing the initialization files",
    )
    parser.add_argument(
        "--parser",
        "-n",
        type=str,
        choices=["character_sheets", "global_outline", "act_outlines", "chapter_outlines"],
        help="Name of the parser to run (optional; if not specified, all parsers are run)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    try:
        results = asyncio.run(run_parser_command(args.project_dir, args.parser))

        # Print results
        print("\nParser Results:")
        print("-" * 50)
        for parser_name, (success, message) in results.items():
            status = "✅" if success else "❌"
            print(f"{status} {parser_name}: {message}")

        # Check if all parsers succeeded
        all_successful = all(success for success, _ in results.values())
        if not all_successful:
            print("\n❌ Some parsers failed")
            exit(1)
        else:
            print("\n✅ All parsers completed successfully")

    except KeyboardInterrupt:
        logger.info("Parser runner shutting down gracefully due to KeyboardInterrupt...")
    except Exception as main_error:
        logger.critical(
            f"Parser runner encountered an unhandled exception: {main_error}",
            exc_info=True,
        )
        print(f"\n❌ Error: {main_error}")
        exit(1)



if __name__ == "__main__":
    main()
