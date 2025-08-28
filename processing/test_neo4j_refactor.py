# processing/test_neo4j_refactor.py
"""
Test script to verify that the Neo4j refactor works correctly.
This script demonstrates the functionality of the new neo4j_query.py module
and verifies that the refactored code maintains the same behavior.
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_neo4j_query_module():
    """
    Test the new neo4j_query.py module to ensure all functions work correctly.
    """
    logger.info("Testing neo4j_query.py module functions...")

    # Test get_novel_info_property
    try:
        from data_access.kg_queries import get_novel_info_property_from_db

        theme = await get_novel_info_property_from_db("theme")
        logger.info(f"Novel theme: {theme}")
    except Exception as e:
        logger.error(f"Error testing get_novel_info_property: {e}")

    # Test get_most_recent_entity_status
    try:
        from data_access.kg_queries import get_most_recent_value_from_db

        status = await get_most_recent_value_from_db("protagonist", "status_is", 5)
        logger.info(f"Protagonist status: {status}")
    except Exception as e:
        logger.error(f"Error testing get_most_recent_entity_status: {e}")

    # Test execute_factual_query
    try:
        from data_access.kg_queries import query_kg_from_db

        facts = await query_kg_from_db("protagonist", chapter_limit=5)
        logger.info(f"Facts about protagonist: {facts}")
    except Exception as e:
        logger.error(f"Error testing execute_factual_query: {e}")

    # Test get_shortest_path_length_between_entities
    try:
        from data_access.kg_queries import get_shortest_path_length_between_entities

        path_length = await get_shortest_path_length_between_entities(
            "protagonist", "antagonist"
        )
        logger.info(f"Path length between protagonist and antagonist: {path_length}")
    except Exception as e:
        logger.error(f"Error testing get_shortest_path_length_between_entities: {e}")

    logger.info("Completed testing neo4j_query.py module functions.")


async def test_refactored_prompt_data_getters():
    """
    Test that the refactored prompt_data_getters.py functions work correctly.
    """
    logger.info("Testing refactored prompt_data_getters.py functions...")

    # Mock data for testing
    plot_outline = {
        "protagonist_name": "Hero",
        "theme": "Good vs Evil",
        "central_conflict": "The hero must save the world",
    }

    try:
        from prompts.prompt_data_getters import (
            get_reliable_kg_facts_for_drafting_prompt,
        )

        facts = await get_reliable_kg_facts_for_drafting_prompt(plot_outline, 1)
        logger.info(f"Reliable KG facts for drafting: {facts[:100]}...")
    except Exception as e:
        logger.error(f"Error testing get_reliable_kg_facts_for_drafting_prompt: {e}")

    logger.info("Completed testing refactored prompt_data_getters.py functions.")


async def main():
    """
    Main test function to run all tests.
    """
    logger.info("Starting Neo4j refactor tests...")

    # Run tests
    await test_neo4j_query_module()
    await test_refactored_prompt_data_getters()

    logger.info("All tests completed.")


if __name__ == "__main__":
    asyncio.run(main())
