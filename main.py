# main.py
import argparse
import asyncio
import logging

from core.database_interface import register_database_services
from core.db_manager import neo4j_manager
from orchestration.nana_orchestrator import NANA_Orchestrator, setup_logging_nana
from core.schema_introspector import SchemaIntrospector
from core.intelligent_type_inference import IntelligentTypeInference
from core.service_registry import register_singleton

schema_introspector = SchemaIntrospector()
register_singleton("schema_introspector", schema_introspector)

type_inference_service = IntelligentTypeInference(schema_introspector)
register_singleton("type_inference_service", type_inference_service)

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging_nana()
    register_database_services()
    from core.schema_introspector import SchemaIntrospector
    from core.intelligent_type_inference import IntelligentTypeInference
    from core.service_registry import register_singleton

    schema_introspector = SchemaIntrospector()
    register_singleton("schema_introspector", schema_introspector)

    type_inference_service = IntelligentTypeInference(schema_introspector)
    register_singleton("type_inference_service", type_inference_service)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", default=None, help="Path to text file to ingest")
    args = parser.parse_args()

    orchestrator = NANA_Orchestrator()
    try:
        if args.ingest:
            asyncio.run(orchestrator.run_ingestion_process(args.ingest))
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
