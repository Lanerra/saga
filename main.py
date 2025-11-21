# main.py
import asyncio

import structlog

from core.db_manager import neo4j_manager
from core.logging_config import setup_saga_logging
from orchestration.langgraph_orchestrator import LangGraphOrchestrator

logger = structlog.get_logger(__name__)


def main() -> None:
    setup_saga_logging()

    orchestrator = LangGraphOrchestrator()

    try:
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
