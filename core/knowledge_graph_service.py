# core/knowledge_graph_service.py
"""
Unified KnowledgeGraph service that handles all KG operations with native models.
Eliminates serialization overhead by working directly with Pydantic models.
"""

import logging
from typing import Any

import structlog

from core.db_manager import neo4j_manager
from data_access.cypher_builders.native_builders import NativeCypherBuilder
from models.kg_models import CharacterProfile, WorldItem

logger = structlog.get_logger(__name__)


class KnowledgeGraphService:
    """Single service handling all KG operations with native models"""
    
    def __init__(self):
        self.cypher_builder = NativeCypherBuilder()
    
    async def persist_entities(
        self, 
        characters: list[CharacterProfile], 
        world_items: list[WorldItem],
        chapter_number: int
    ) -> bool:
        """
        Persist entities directly from models without dict conversion.
        
        Args:
            characters: List of CharacterProfile models
            world_items: List of WorldItem models  
            chapter_number: Current chapter for tracking
            
        Returns:
            True if successful, False otherwise
        """
        statements: list[tuple[str, dict[str, Any]]] = []
        
        try:
            # Generate Cypher directly from models
            for char in characters:
                cypher, params = self.cypher_builder.character_upsert_cypher(
                    char, chapter_number
                )
                statements.append((cypher, params))
            
            for item in world_items:
                cypher, params = self.cypher_builder.world_item_upsert_cypher(
                    item, chapter_number
                )
                statements.append((cypher, params))
            
            if statements:
                await neo4j_manager.execute_cypher_batch(statements)
            
            logger.info(
                "Persisted %d characters and %d world items for chapter %d using native models",
                len(characters),
                len(world_items), 
                chapter_number
            )
            return True
            
        except Exception as exc:
            logger.error(
                "Error persisting entities for chapter %d: %s",
                chapter_number,
                exc,
                exc_info=True,
            )
            return False
    
    async def fetch_characters(self, filters: dict[str, Any] = None) -> list[CharacterProfile]:
        """
        Fetch characters directly as models without dict conversion.
        
        Args:
            filters: Optional filters for the query
            
        Returns:
            List of CharacterProfile models
        """
        try:
            query = """
            MATCH (c:Character:Entity)
            WHERE c.is_deleted IS NULL OR c.is_deleted = FALSE
            RETURN c
            ORDER BY c.name
            """
            
            results = await neo4j_manager.execute_read_query(query, filters or {})
            characters = []
            
            for record in results:
                if record and record.get('c'):
                    char = CharacterProfile.from_db_record(record)
                    characters.append(char)
            
            logger.debug("Fetched %d characters using native models", len(characters))
            return characters
            
        except Exception as exc:
            logger.error("Error fetching characters: %s", exc, exc_info=True)
            return []
    
    async def fetch_world_items(self, filters: dict[str, Any] = None) -> list[WorldItem]:
        """
        Fetch world items directly as models without dict conversion.
        
        Args:
            filters: Optional filters for the query
            
        Returns:
            List of WorldItem models
        """
        try:
            query = """
            MATCH (w:WorldElement:Entity)
            WHERE w.is_deleted IS NULL OR w.is_deleted = FALSE
            RETURN w
            ORDER BY w.category, w.name
            """
            
            results = await neo4j_manager.execute_read_query(query, filters or {})
            world_items = []
            
            for record in results:
                if record and record.get('w'):
                    item = WorldItem.from_db_record(record)
                    world_items.append(item)
            
            logger.debug("Fetched %d world items using native models", len(world_items))
            return world_items
            
        except Exception as exc:
            logger.error("Error fetching world items: %s", exc, exc_info=True)
            return []
    
    async def fetch_entities_for_context(
        self,
        chapter_number: int,
        character_limit: int = 10,
        world_item_limit: int = 10
    ) -> tuple[list[CharacterProfile], list[WorldItem]]:
        """
        Fetch entities relevant for context generation without conversion overhead.
        
        Args:
            chapter_number: Current chapter being processed
            character_limit: Max characters to return
            world_item_limit: Max world items to return
            
        Returns:
            Tuple of (characters, world_items)
        """
        try:
            # Single query to get both characters and world items with relevance
            query = """
            // Get characters that appeared in recent chapters
            MATCH (c:Character:Entity)-[:APPEARS_IN]->(ch:Chapter)
            WHERE ch.number < $chapter_number
            WITH c, max(ch.number) as last_appearance
            ORDER BY last_appearance DESC
            LIMIT $character_limit
            
            WITH collect({type: 'character', node: c}) as character_nodes
            
            // Get world items referenced in recent chapters  
            MATCH (w:WorldElement:Entity)-[:REFERENCED_IN]->(ch:Chapter)
            WHERE ch.number < $chapter_number
            WITH character_nodes, w, max(ch.number) as last_reference
            ORDER BY last_reference DESC
            LIMIT $world_item_limit
            
            WITH character_nodes, collect({type: 'world', node: w}) as world_nodes
            
            RETURN character_nodes + world_nodes as entities
            """
            
            results = await neo4j_manager.execute_read_query(
                query,
                {
                    "chapter_number": chapter_number,
                    "character_limit": character_limit,
                    "world_item_limit": world_item_limit
                }
            )
            
            characters = []
            world_items = []
            
            if results and results[0]:
                entities = results[0].get('entities', [])
                for entity_data in entities:
                    if entity_data['type'] == 'character':
                        char = CharacterProfile.from_db_node(entity_data['node'])
                        characters.append(char)
                    elif entity_data['type'] == 'world':
                        item = WorldItem.from_db_node(entity_data['node'])
                        world_items.append(item)
            
            logger.debug(
                "Fetched %d characters and %d world items for context (chapter %d)",
                len(characters), len(world_items), chapter_number
            )
            
            return characters, world_items
            
        except Exception as exc:
            logger.error(
                "Error fetching entities for context (chapter %d): %s", 
                chapter_number, exc, exc_info=True
            )
            return [], []


# Singleton instance
knowledge_graph_service = KnowledgeGraphService()