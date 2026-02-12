# main.py
import argparse
import asyncio
import sys
from pathlib import Path

import structlog

from core.db_manager import neo4j_manager
from core.llm_interface_refactored import async_llm_context
from core.logging_config import setup_saga_logging
from core.parser_runner import run_parser_command
from core.project_bootstrapper import ProjectBootstrapper
from core.project_config import NarrativeProjectConfig
from core.project_manager import ProjectManager
from orchestration.langgraph_orchestrator import LangGraphOrchestrator

logger = structlog.get_logger(__name__)


async def run_bootstrap_mode(user_prompt: str, *, review: bool) -> tuple[Path, NarrativeProjectConfig]:
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("Bootstrap mode requires a non-empty prompt")

    async with async_llm_context() as (language_model_service, _embedding_service):
        bootstrapper = ProjectBootstrapper(language_model_service)
        project_config = await bootstrapper.generate_metadata(user_prompt.strip())
        project_directory = bootstrapper.save_config(project_config, review=review)

    logger.info(
        "Bootstrap completed",
        project_dir=str(project_directory),
        title=project_config.title,
        review=review,
    )
    return project_directory, project_config


async def run_generation_mode(*, from_candidate: bool) -> None:
    if from_candidate:
        project_directory = ProjectManager.find_candidate_project()
        if project_directory is None:
            raise ValueError("No candidate config found for generate mode with --from-candidate flag")
        ProjectManager.promote_candidate(project_directory)
    else:
        candidate_directory = ProjectManager.find_candidate_project()
        if candidate_directory is not None:
            logger.info("Found candidate config, using it for generation")
            ProjectManager.promote_candidate(candidate_directory)
            project_directory = candidate_directory
        else:
            project_directory = ProjectManager.find_resume_project()
            if project_directory is None:
                project_directory = ProjectManager.create_default_project()

    project_config = ProjectManager.load_config(project_directory)
    orchestrator = LangGraphOrchestrator(project_dir=project_directory)
    await orchestrator.run_novel_generation_loop(narrative_config=project_config)


async def run_quick_mode(user_prompt: str) -> None:
    project_directory, project_config = await run_bootstrap_mode(user_prompt, review=False)
    orchestrator = LangGraphOrchestrator(project_dir=project_directory)
    await orchestrator.run_novel_generation_loop(narrative_config=project_config)


def main() -> None:
    setup_saga_logging()

    parser = argparse.ArgumentParser(description="SAGA - Autonomous Novel Generation")
    subparsers = parser.add_subparsers(title="Commands", dest="command")

    # Quick mode (default)
    quick_parser = subparsers.add_parser("quick", help="Quick mode: bootstrap and generate")
    quick_parser.add_argument("prompt", help="Story premise")

    # Bootstrap mode
    bootstrap_parser = subparsers.add_parser("bootstrap", help="Bootstrap mode: generate metadata")
    bootstrap_parser.add_argument("prompt", help="Story premise")

    # Generate mode
    generate_parser = subparsers.add_parser("generate", help="Generate mode: run novel generation loop")
    generate_parser.add_argument(
        "--from-candidate",
        action="store_true",
        help="Use candidate config instead of existing project",
    )

    # Parser mode
    parser_parser = subparsers.add_parser("parse", help="Parser mode: run parsers independently")
    parser_parser.add_argument(
        "--project-dir",
        "-p",
        type=str,
        help="Path to the project directory containing the initialization files",
    )
    parser_parser.add_argument(
        "--parser",
        "-n",
        type=str,
        choices=["character_sheets", "global_outline", "act_outlines", "chapter_outlines"],
        help="Name of the parser to run (optional; if not specified, all parsers are run)",
    )

    arguments = parser.parse_args()

    try:
        if arguments.command == "bootstrap":
            asyncio.run(run_bootstrap_mode(arguments.prompt, review=True))
        elif arguments.command == "generate":
            asyncio.run(run_generation_mode(from_candidate=arguments.from_candidate))
        elif arguments.command == "quick":
            asyncio.run(run_quick_mode(arguments.prompt))
        elif arguments.command == "parse":
            results = asyncio.run(run_parser_command(arguments.project_dir, arguments.parser))
            # Print results
            print("\nParser Results:")
            print("-" * 50)
            for parser_name, (success, message) in results.items():
                status = "✅" if success else "❌"
                print(f"{status} {parser_name}: {message}")
            # Check if all parsers succeeded
            all_successful = all(success for success, _ in results.values())
            if not all_successful:
                exit(1)
        elif arguments.command:
            parser.error(f"Unknown command: {arguments.command}")
        else:
            # Default to generate mode
            asyncio.run(run_generation_mode(from_candidate=False))
    except KeyboardInterrupt:
        logger.info("SAGA Orchestrator shutting down gracefully due to KeyboardInterrupt...")
    except Exception as main_error:  # pragma: no cover - entry point catch
        logger.critical(
            f"SAGA Orchestrator encountered an unhandled main exception: {main_error}",
            exc_info=True,
        )
        sys.exit(1)
    finally:
        if neo4j_manager.driver is not None:
            logger.info("Ensuring Neo4j driver is closed from main entry point.")

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
                    f"Could not explicitly close driver from main: {close_error}",
                )


if __name__ == "__main__":
    main()
