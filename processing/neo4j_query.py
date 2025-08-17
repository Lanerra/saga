# processing/neo4j_query.py
"""
Consolidated Neo4j query functions to eliminate duplicate implementations
and follow the DRY principle. This module provides a unified interface
for common Neo4j operations used throughout the SAGA system.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from core.llm_interface import llm_service
from data_access import chapter_queries, kg_queries

logger = logging.getLogger(__name__)


async def execute_vector_search(query_text: str, chapter_number: int, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Execute semantic search with vector index for chapter context.
    
    Args:
        query_text: Text to use for semantic search
        chapter_number: Current chapter number (excluded from results)
        limit: Maximum number of similar chapters to return
        
    Returns:
        List of similar chapters with their data
    """
    logger.debug(f"Executing vector search for chapter {chapter_number} context...")
    
    # Get embedding for the query text
    query_embedding = await llm_service.async_get_embedding(query_text)
    if query_embedding is None:
        logger.warning("Failed to generate embedding for semantic context query.")
        return []
    
    # Find similar chapters using the embedding
    similar_chapters = await chapter_queries.find_similar_chapters_in_db(
        query_embedding, limit, chapter_number
    )
    
    return similar_chapters


async def execute_factual_query(subject: str, predicate: Optional[str] = None, 
                              chapter_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Execute factual data retrieval from the knowledge graph.
    
    Args:
        subject: Subject entity name
        predicate: Relationship type (optional)
        chapter_limit: Maximum chapter number to consider (optional)
        
    Returns:
        List of factual relationships
    """
    logger.debug(f"Executing factual query for subject '{subject}'...")
    
    results = await kg_queries.query_kg_from_db(
        subject=subject,
        predicate=predicate,
        chapter_limit=chapter_limit,
        include_provisional=False,
        limit_results=10
    )
    
    return results


async def get_most_recent_entity_status(subject: str, predicate: str, 
                                      chapter_limit: Optional[int] = None) -> Optional[Any]:
    """
    Get the most recent value for a specific entity property.
    
    Args:
        subject: Subject entity name
        predicate: Property/relationship type
        chapter_limit: Maximum chapter number to consider
        
    Returns:
        Most recent value or None if not found
    """
    logger.debug(f"Getting most recent status for '{subject}' -> '{predicate}'...")
    
    value = await kg_queries.get_most_recent_value_from_db(
        subject=subject,
        predicate=predicate,
        chapter_limit=chapter_limit,
        include_provisional=False
    )
    
    return value


async def get_novel_info_property(property_key: str) -> Optional[Any]:
    """
    Get a property value from the NovelInfo node.
    
    Args:
        property_key: Name of the property to retrieve
        
    Returns:
        Property value or None if not found
    """
    logger.debug(f"Getting novel info property '{property_key}'...")
    
    value = await kg_queries.get_novel_info_property_from_db(property_key)
    return value


async def get_chapter_context_for_entity(entity_name: Optional[str] = None, 
                                       entity_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get chapter context for an entity from the knowledge graph.
    
    Args:
        entity_name: Name of the entity (for Characters/ValueNodes)
        entity_id: ID of the entity (for WorldElements)
        
    Returns:
        List of chapters where the entity was mentioned
    """
    logger.debug(f"Getting chapter context for entity '{entity_name or entity_id}'...")
    
    results = await kg_queries.get_chapter_context_for_entity(
        entity_name=entity_name,
        entity_id=entity_id
    )
    
    return results


async def get_entity_context_for_resolution(entity_id: str) -> Optional[Dict[str, Any]]:
    """
    Get comprehensive context for an entity to help with resolution decisions.
    
    Args:
        entity_id: ID of the entity to get context for
        
    Returns:
        Dictionary with entity context or None if not found
    """
    logger.debug(f"Getting entity context for resolution of entity '{entity_id}'...")
    
    context = await kg_queries.get_entity_context_for_resolution(entity_id)
    return context


async def get_shortest_path_length_between_entities(name1: str, name2: str, max_depth: int = 4) -> Optional[int]:
    """
    Get the shortest path length between two entities.
    
    Args:
        name1: Name of the first entity
        name2: Name of the second entity
        max_depth: Maximum depth to search
        
    Returns:
        Shortest path length or None if no path found
    """
    logger.debug(f"Getting shortest path length between '{name1}' and '{name2}'...")
    
    path_len = await kg_queries.get_shortest_path_length_between_entities(
        name1, name2, max_depth
    )
    return path_len