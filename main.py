# main.py
import argparse
import asyncio
import sys
from pathlib import Path

import structlog

from core.langgraph.export import generate_full_export
from core.logging_config import setup_saga_logging
from core.parser_runner import run_parser_command
from core.project_bootstrapper import ProjectBootstrapper
from core.project_config import NarrativeProjectConfig
from core.project_manager import ProjectManager
from core.service_context import RunServices, service_lifetime
from orchestration.langgraph_orchestrator import LangGraphOrchestrator

logger = structlog.get_logger(__name__)


async def run_bootstrap_mode(user_prompt: str, *, review: bool, services: RunServices | None = None) -> tuple[Path, NarrativeProjectConfig]:
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("Bootstrap mode requires a non-empty prompt")

    async with service_lifetime(services) as run_services:
        bootstrapper = ProjectBootstrapper(run_services.language_model)
        project_config = await bootstrapper.generate_metadata(user_prompt.strip())
        project_directory = bootstrapper.save_config(project_config, review=review)

    logger.info(
        "Bootstrap completed",
        project_dir=str(project_directory),
        title=project_config.title,
        review=review,
    )
    config_name = "config.candidate.json" if review else "config.json"
    print(f"SAGA bootstrap succeeded: {project_directory / config_name}")
    if review:
        print(f'Review this candidate, then run: python main.py generate --project-dir "{project_directory}" --from-candidate')
    return project_directory, project_config


async def run_generation_mode(*, project_directory: Path, from_candidate: bool, services: RunServices | None = None) -> None:
    if not project_directory.is_dir():
        raise FileNotFoundError(f"Project directory does not exist: {project_directory}")
    print(f"Selected project: {project_directory.absolute()}")
    print("Scope: resume the selected project's durable workflow, or initialize only if fresh. No reset or deletion is requested.")
    print(f"Checkpoint: {project_directory.absolute() / 'checkpoints/saga.db'}; graph ownership and retained artifacts are reconciled before continuation.")
    if from_candidate:
        ProjectManager.promote_candidate(project_directory)

    project_config = ProjectManager.load_config(project_directory)
    orchestrator = LangGraphOrchestrator(project_dir=project_directory, services=services)
    await orchestrator.run_novel_generation_loop(narrative_config=project_config)
    completed = ProjectManager.count_completed_chapters(project_directory)
    print(f"SAGA generation invocation succeeded: {project_directory.absolute()}; accepted manuscripts {completed}/{project_config.total_chapters}. Export is a separate command.")


async def run_quick_mode(user_prompt: str, *, services: RunServices | None = None) -> None:
    async with service_lifetime(services) as run_services:
        project_directory, _ = await run_bootstrap_mode(user_prompt, review=False, services=run_services)
        await run_generation_mode(project_directory=project_directory, from_candidate=False, services=run_services)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGA - Autonomous Novel Generation", epilog="Generation resumes only the explicitly selected project. No command here resets checkpoints, graph data or author files.")
    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    # Quick mode
    quick_parser = subparsers.add_parser("quick", help="Quick mode: bootstrap and generate")
    quick_parser.add_argument("prompt", help="Story premise")

    # Bootstrap mode
    bootstrap_parser = subparsers.add_parser("bootstrap", help="Bootstrap mode: generate metadata")
    bootstrap_parser.add_argument("prompt", help="Story premise")

    # Generate mode
    generate_parser = subparsers.add_parser("generate", help="Generate mode: run novel generation loop")
    generate_parser.add_argument("--project-dir", "-p", type=Path, required=True, help="Exact project to initialize or resume; no reset or deletion")
    generate_parser.add_argument(
        "--from-candidate",
        action="store_true",
        help="Validate and promote this project's config.candidate.json; never replace config.json",
    )

    # Parser mode
    parser_parser = subparsers.add_parser("parse", help="Accept a complete retained initialization import; individual parser writes are blocked")
    parser_parser.add_argument(
        "--project-dir",
        "-p",
        type=str,
        required=True,
        help="Path to the project directory containing the initialization files",
    )
    parser_parser.add_argument(
        "--parser",
        "-n",
        type=str,
        choices=["character_sheets", "global_outline", "act_outlines", "chapter_outlines"],
        help="Name of the parser to run (optional; if not specified, all parsers are run)",
    )

    export_parser = subparsers.add_parser("export", help="Export every configured chapter from checksum-verified accepted manuscripts; no drafts")
    export_parser.add_argument("--project-dir", "-p", type=Path, required=True, help="Exact project containing config.json and accepted manuscript receipts")

    arguments = parser.parse_args()
    setup_saga_logging()

    try:
        if arguments.command == "bootstrap":
            asyncio.run(run_bootstrap_mode(arguments.prompt, review=True))
        elif arguments.command == "generate":
            asyncio.run(run_generation_mode(project_directory=arguments.project_dir, from_candidate=arguments.from_candidate))
        elif arguments.command == "quick":
            asyncio.run(run_quick_mode(arguments.prompt))
        elif arguments.command == "export":
            project_directory = arguments.project_dir.absolute()
            project_config = ProjectManager.load_config(project_directory)
            output = generate_full_export(project_directory, expected_chapters=project_config.total_chapters)
            chapters = ", ".join(str(number) for number in range(1, project_config.total_chapters + 1))
            print(f"SAGA export succeeded: chapters {chapters} -> {output}")
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
                print("SAGA parse failed; retained artifacts may require reconciliation.", file=sys.stderr)
                sys.exit(1)
            print("SAGA parse succeeded: complete retained initialization accepted.")
        elif arguments.command:
            parser.error(f"Unknown command: {arguments.command}")
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("SAGA generation cancelled")
        print(f"SAGA {arguments.command} cancelled; no completion claimed. Retained artifacts may include partial progress; resume the same project without resetting.", file=sys.stderr)
        sys.exit(130)
    except Exception as main_error:  # pragma: no cover - entry point catch
        logger.critical(
            f"SAGA Orchestrator encountered an unhandled main exception: {main_error}",
            exc_info=True,
        )
        print(f"SAGA {arguments.command} failed: {main_error}. No completion claimed; retained artifacts may include partial progress.", file=sys.stderr)
        sys.exit(1)



if __name__ == "__main__":
    main()
