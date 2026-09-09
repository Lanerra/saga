#!/usr/bin/env python3
"""Explicit acceptance/recovery CLI for a retained complete initialization import."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

import structlog

from core.logging_config import setup_saga_logging
from core.parsers.act_outline_parser import ActOutlineParser
from core.parsers.chapter_outline_parser import ChapterOutlineParser
from core.parsers.character_sheet_parser import CharacterSheetParser
from core.parsers.global_outline_parser import GlobalOutlineParser
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
    """Accept/recover a complete frozen import in an explicitly selected project.

    Args:
        project_dir_path: Nonempty path to an existing project, without traversal or symlinks.
        parser_name: Legacy individual parser selector; these writes are blocked.

    Returns:
        Complete initialization acceptance result, or an individual-parser refusal.
    """
    if not isinstance(project_dir_path, str) or not project_dir_path.strip():
        raise ValueError("Select an explicit nonempty --project-dir")
    resolved_project_dir = Path(project_dir_path).absolute()
    if ".." in resolved_project_dir.parts:
        raise ValueError("Project directory must not contain parent traversal")
    if any(path.is_symlink() for path in (*resolved_project_dir.parents, resolved_project_dir)):
        raise ValueError("Project directory must not contain symbolic links")
    if not resolved_project_dir.is_dir():
        raise FileNotFoundError(f"Project directory does not exist: {resolved_project_dir}")

    runner = ParserRunner(resolved_project_dir)
    if parser_name is not None:
        success, message = await runner.run_parser(parser_name)
        return {parser_name: (success, message)}

    setup_saga_logging()
    logger.info("Accepting complete frozen initialization import", project_dir=str(resolved_project_dir))
    async with service_lifetime(services):
        return await runner.run_all_parsers()


def main() -> None:
    """Main entry point for the parser runner CLI."""
    parser = argparse.ArgumentParser(
        description="Accept or recover a complete frozen initialization import in an explicitly selected project.",
        epilog="Individual parser writes are blocked. No project discovery, initialization generation, reset or deletion is requested.",
    )
    parser.add_argument(
        "--project-dir",
        "-p",
        type=str,
        required=True,
        help="Existing project containing the selected frozen initialization import",
    )
    parser.add_argument(
        "--parser",
        "-n",
        type=str,
        choices=["character_sheets", "global_outline", "act_outlines", "chapter_outlines"],
        help="Legacy selector: individual parser writes are blocked; omit to accept the complete frozen import",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Legacy compatibility flag; logging follows configured settings",
    )

    args = parser.parse_args()

    try:
        results = asyncio.run(run_parser_command(args.project_dir, args.parser))

        print("\nInitialization Acceptance Results:")
        print("-" * 50)
        for parser_name, (success, message) in results.items():
            status = "✅" if success else "❌"
            print(f"{status} {parser_name}: {message}")

        all_successful = all(success for success, _ in results.values())
        if not all_successful:
            print("\nSAGA initialization acceptance failed; retained artifacts may require reconciliation.", file=sys.stderr)
            sys.exit(1)
        else:
            print("\nSAGA initialization acceptance succeeded: complete retained initialization accepted.")

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("SAGA initialization acceptance cancelled; no completion claimed. Retained artifacts may include partial progress; recover the same project without resetting.", file=sys.stderr)
        sys.exit(130)
    except Exception as main_error:
        logger.critical(
            f"Parser runner encountered an unhandled exception: {main_error}",
            exc_info=True,
        )
        print(f"\nSAGA initialization acceptance failed: {main_error}. No completion claimed.", file=sys.stderr)
        sys.exit(1)



if __name__ == "__main__":
    main()
