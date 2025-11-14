# main.py
import argparse
import asyncio

import structlog

from core.db_manager import neo4j_manager
from core.logging_config import setup_saga_logging
from initialization.bootstrap_pipeline import run_bootstrap_pipeline
from orchestration.langgraph_orchestrator import LangGraphOrchestrator

logger = structlog.get_logger(__name__)


def main() -> None:
    setup_saga_logging()

    parser = argparse.ArgumentParser()

    # Bootstrap CLI flags (standalone and integrated)
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Run the multi-phase bootstrap pipeline and exit.",
    )
    parser.add_argument(
        "--bootstrap-phase",
        choices=["world", "characters", "plot", "all"],
        default="all",
        help="Limit bootstrap to a specific phase.",
    )
    parser.add_argument(
        "--bootstrap-level",
        choices=["basic", "enhanced", "max"],
        default="enhanced",
        help="Control target counts and validation strictness.",
    )
    parser.add_argument(
        "--bootstrap-dry-run",
        action="store_true",
        help="Generate and validate but do not write to Neo4j.",
    )
    parser.add_argument(
        "--bootstrap-reset-kg",
        action="store_true",
        help="Reset the Knowledge Graph before bootstrapping.",
    )
    args = parser.parse_args()

    orchestrator = LangGraphOrchestrator()

    try:
        if args.bootstrap:
            # Optional reset wrapper using existing reset script
            if args.bootstrap_reset_kg:
                from reset_neo4j import reset_neo4j_database_async

                asyncio.run(
                    reset_neo4j_database_async(
                        uri=None, user=None, password=None, confirm=True
                    )
                )

            # Ensure DB connection/schema before KG writes
            asyncio.run(neo4j_manager.connect())
            asyncio.run(neo4j_manager.create_db_schema())

            # Run the bootstrap pipeline per proposal
            plot_outline, character_profiles, world_building, warnings = asyncio.run(
                run_bootstrap_pipeline(
                    phase=args.bootstrap_phase,  # type: ignore[arg-type]
                    level=args.bootstrap_level,  # type: ignore[arg-type]
                    dry_run=args.bootstrap_dry_run,
                )
            )
            logger.info(
                "Bootstrap complete. Title: '%s'; Plot points: %d; Characters: %d; World cats: %d",
                plot_outline.get("title", "(unknown)"),
                len(
                    plot_outline.get("plot_points", [])
                    if isinstance(plot_outline.get("plot_points"), list)
                    else []
                ),
                len(character_profiles),
                len(
                    [
                        k
                        for k in world_building.keys()
                        if k not in ("is_default", "source")
                    ]
                ),
            )
            if warnings:
                logger.warning("Bootstrap warnings: %s", "; ".join(warnings))
        else:
            asyncio.run(orchestrator.run_novel_generation_loop())
    except KeyboardInterrupt:
        logger.info(
            "SAGA Orchestrator shutting down gracefully due to KeyboardInterrupt..."
        )
    except Exception as main_err:  # pragma: no cover - entry point catch
        logger.critical(
            f"SAGA Orchestrator encountered an unhandled main exception: {main_err}",
            exc_info=True,
        )
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
            except Exception as e:
                logger.warning(
                    f"Could not explicitly close driver from main: {e}",
                )


if __name__ == "__main__":
    main()
