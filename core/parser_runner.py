#!/usr/bin/env python3
"""CLI command to run parsers independently from SAGA initialization phase.

This module provides a command-line interface for running parsers independently
for testing and debugging purposes. It supports running all parsers sequentially
or individual parsers by name.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import structlog

from core.db_manager import neo4j_manager
from core.logging_config import setup_saga_logging
from core.parsers.act_outline_parser import ActOutlineParser
from core.parsers.chapter_outline_parser import ChapterOutlineParser
from core.parsers.character_sheet_parser import CharacterSheetParser
from core.parsers.global_outline_parser import GlobalOutlineParser
from core.project_manager import ProjectManager

logger = structlog.get_logger(__name__)


class ParserRunner:
    """Coordinates running parsers independently from the initialization phase."""

    def __init__(self, project_dir: Path):
        """Initialize the parser runner.

        Args:
            project_dir: Path to the project directory containing the initialization files.
        """
        self.project_dir = project_dir
        from core.langgraph.content_manager import ContentManager
        self.content_manager = ContentManager(str(project_dir))

    async def run_all_parsers(self) -> dict[str, tuple[bool, str]]:
        """Run all parsers in the correct order.

        Returns:
            Dictionary containing results for each parser.
        """
        results = {}

        # Run parsers in order (Stage 1 -> Stage 2 -> Stage 3 -> Stage 4)
        parsers = [
            ("character_sheets", CharacterSheetParser),
            ("global_outline", GlobalOutlineParser),
            ("act_outlines", ActOutlineParser),
            ("chapter_outlines", ChapterOutlineParser),
        ]

        for parser_name, parser_class in parsers:
            logger.info("Running parser", parser=parser_name)
            success, message = await self._run_parser(parser_class)
            results[parser_name] = (success, message)

            if not success:
                logger.error("Parser failed", parser=parser_name, message=message)

        return results

    async def run_parser(self, parser_name: str) -> tuple[bool, str]:
        """Run a specific parser by name.

        Args:
            parser_name: Name of the parser to run.

        Returns:
            Tuple of (success: bool, message: str).

        Raises:
            ValueError: If parser name is unknown.
        """
        parser_map = {
            "character_sheets": CharacterSheetParser,
            "global_outline": GlobalOutlineParser,
            "act_outlines": ActOutlineParser,
            "chapter_outlines": ChapterOutlineParser,
        }

        if parser_name not in parser_map:
            raise ValueError(f"Unknown parser: {parser_name}")

        logger.info("Running parser", parser=parser_name)
        return await self._run_parser(parser_map[parser_name])

    async def _run_parser(self, parser_class: type) -> tuple[bool, str]:
        """Run a specific parser instance.

        Args:
            parser_class: The parser class to instantiate and run.

        Returns:
            Tuple of (success: bool, message: str).
        """
        try:
            # Get default file path for the parser
            parser = self._create_parser_instance(parser_class)
            return await parser.parse_and_persist()
        except Exception as e:
            logger.error("Error running parser", error=str(e), exc_info=True)
            return False, str(e)

    def _create_parser_instance(self, parser_class: type) -> Any:
        """Create a parser instance with the correct file path from ContentManager.

        Args:
            parser_class: The parser class to instantiate.

        Returns:
            Parser instance.
        """
        if parser_class == CharacterSheetParser:
            # Get character sheets from .saga/content/character_sheets/all_v1.json
            file_path = self.content_manager._get_content_path("character_sheets", "all", 1, "json")
            return CharacterSheetParser(character_sheets_path=str(file_path))
        elif parser_class == GlobalOutlineParser:
            # Get global outline from .saga/content/global_outline/main_v1.json
            file_path = self.content_manager._get_content_path("global_outline", "main", 1, "json")
            return GlobalOutlineParser(global_outline_path=str(file_path))
        elif parser_class == ActOutlineParser:
            # Get act outlines from .saga/content/act_outlines/all_v1.json
            file_path = self.content_manager._get_content_path("act_outlines", "all", 1, "json")
            return ActOutlineParser(act_outline_path=str(file_path))
        elif parser_class == ChapterOutlineParser:
            v0_path = self.content_manager._get_content_path("chapter_outlines", "all", 0, "json")
            v1_path = self.content_manager._get_content_path("chapter_outlines", "all", 1, "json")

            if v0_path.exists():
                file_path = v0_path
            elif v1_path.exists():
                file_path = v1_path
            else:
                file_path = v1_path

            return ChapterOutlineParser(chapter_outline_path=str(file_path))
        else:
            raise ValueError(f"Unsupported parser class: {parser_class}")


async def run_parser_command(
    project_dir: str | None, parser_name: str | None
) -> dict[str, tuple[bool, str]]:
    """Run the parser command with the given arguments.

    Args:
        project_dir: Path to the project directory. If None, use default.
        parser_name: Name of the parser to run. If None, run all parsers.

    Returns:
        Dictionary containing results for each parser.
    """
    setup_saga_logging()

    if project_dir is None:
        project_dir = ProjectManager.find_resume_project()
        if project_dir is None:
            project_dir = ProjectManager.create_default_project()
    else:
        project_dir = Path(project_dir)

    logger.info("Running parsers", project_dir=str(project_dir), parser=parser_name)

    runner = ParserRunner(project_dir)

    if parser_name:
        success, message = await runner.run_parser(parser_name)
        return {parser_name: (success, message)}
    else:
        return await runner.run_all_parsers()


def main() -> None:
    """Main entry point for the parser runner CLI."""
    parser = argparse.ArgumentParser(
        description="Run SAGA parsers independently for testing and debugging purposes"
    )
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
    finally:
        if neo4j_manager.driver is not None:
            logger.info("Ensuring Neo4j driver is closed from parser runner entry point.")

            async def _close_driver_main() -> None:
                await neo4j_manager.close()

            try:
                loop = asyncio.get_running_loop()
                if not loop.is_closed():
                    loop.create_task(_close_driver_main())
            except RuntimeError:
                asyncio.run(_close_driver_main())
            except Exception as close_error:
                logger.warning(
                    f"Could not explicitly close driver from parser runner: {close_error}",
                )


if __name__ == "__main__":
    main()
