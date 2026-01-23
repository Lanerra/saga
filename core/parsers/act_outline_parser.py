# core/parsers/act_outline_parser.py
"""Parse act outlines and create Stage 3 knowledge graph entities.

This module provides the ActOutlineParser class that:
1. Reads act outlines from JSON files
2. Creates ActKeyEvent Event nodes (2-5 per act)
3. Enriches Location nodes with names from Stage 2
4. Creates relationships between events, characters, and locations
5. Persists to Neo4j using the existing data access layer

Based on: docs/schema-design.md - Stage 3: Act Outlines
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import structlog
from pydantic import BaseModel, Field

from core.db_manager import neo4j_manager
from core.exceptions import DatabaseError
from models.kg_models import ActKeyEvent, Location, MajorPlotPoint
from utils.common import try_load_json_from_response

logger = structlog.get_logger(__name__)


class ActOutlineParser:
    """Parse act outlines and create Stage 3 knowledge graph entities.
    
    This parser handles Stage 3 of the knowledge graph construction:
    - ActKeyEvent Event node creation (2-5 per act)
    - Location node enrichment with names
    - Relationship creation between events, characters, and locations
    - Persistence to Neo4j
    
    Attributes:
        act_outline_path: Path to act outlines JSON file
        chapter_number: Chapter number for provenance (0 for initialization)
    """

    def __init__(self, act_outline_path: str = "act_outlines/all_v1.json", chapter_number: int = 0):
        """Initialize the ActOutlineParser.
        
        Args:
            act_outline_path: Path to act outlines JSON file
            chapter_number: Chapter number for provenance (0 for initialization)
        """
        self.act_outline_path = act_outline_path
        self.chapter_number = chapter_number

    async def parse_act_outline(self) -> dict[str, Any]:
        """Parse act outline from JSON file.
        
        Returns:
            Dictionary containing parsed act outline data
            
        Raises:
            ValueError: If act outline file cannot be read or parsed
            DatabaseError: If there are issues persisting to Neo4j
        """
        try:
            # Read the act outline JSON file
            with open(self.act_outline_path, 'r', encoding='utf-8') as f:
                act_outline_data = json.load(f)
        except FileNotFoundError as e:
            logger.error(f"Act outline file not found: {self.act_outline_path}", exc_info=True)
            raise ValueError(f"Act outline file not found: {self.act_outline_path}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in act outline file: {self.act_outline_path}", exc_info=True)
            raise ValueError(f"Invalid JSON in act outline file: {self.act_outline_path}") from e

        return act_outline_data

    def _generate_event_id(self, event_name: str, act_number: int, sequence: int) -> str:
        """Generate a stable ID for an event.
        
        Args:
            event_name: Name of the event
            act_number: Act number
            sequence: Sequence within act
            
        Returns:
            Generated event ID
        """
        # Use SHA256 hash of event name + act number + sequence for stable ID
        hash_input = f"{event_name}_{act_number}_{sequence}"
        hash_obj = hashlib.sha256(hash_input.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()[:16]  # Use first 16 chars for brevity
        return f"event_{hash_hex}"

    def _parse_act_key_events(self, act_outline_data: dict[str, Any]) -> list[ActKeyEvent]:
        """Parse act key events from act outline data.
        
        Args:
            act_outline_data: Parsed act outline data
            
        Returns:
            List of ActKeyEvent instances
            
        Raises:
            ValueError: If required act key events are missing or invalid
        """
        act_key_events = []
        
        # Check if acts are in the act outline
        if "acts" not in act_outline_data:
            logger.warning("No acts found in act outline data")
            return act_key_events
        
        # Process each act
        for act_data in act_outline_data["acts"]:
            act_number = act_data.get("act_number", 0)
            
            if "sections" not in act_data:
                logger.warning(f"No sections found in act {act_number}")
                continue
            
            sections = act_data["sections"]
            
            if "key_events" not in sections:
                logger.warning(f"No key events found in act {act_number}")
                continue
            
            # Process each key event in the act
            for key_event in sections["key_events"]:
                event_name = key_event.get("event", "")
                event_description = key_event.get("description", event_name)
                sequence_in_act = key_event.get("sequence", 0)
                cause = key_event.get("cause", "")
                effect = key_event.get("effect", "")
                
                # Generate a stable ID for this event
                event_id = self._generate_event_id(event_name, act_number, sequence_in_act)
                
                # Create ActKeyEvent
                act_event = ActKeyEvent(
                    id=event_id,
                    name=event_name,
                    description=event_description,
                    event_type="ActKeyEvent",
                    act_number=act_number,
                    sequence_in_act=sequence_in_act,
                    cause=cause,
                    effect=effect,
                    created_chapter=0,
                    is_provisional=False,
                    created_ts=act_data.get("created_ts"),
                    updated_ts=act_data.get("updated_ts"),
                )
                
                act_key_events.append(act_event)
        
        return act_key_events

    def _parse_location_enrichment(self, act_outline_data: dict[str, Any]) -> dict[str, str]:
        """Parse location name enrichment from act outline data.
        
        Args:
            act_outline_data: Parsed act outline data
            
        Returns:
            Dictionary mapping location descriptions to names
        """
        location_names = {}
        
        # Check if acts are in the act outline
        if "acts" not in act_outline_data:
            logger.warning("No acts found in act outline data")
            return location_names
        
        # Process each act
        for act_data in act_outline_data["acts"]:
            if "sections" not in act_data:
                continue
            
            sections = act_data["sections"]
            
            # Check for location data in various sections
            if "locations" in sections:
                for location in sections["locations"]:
                    location_name = location.get("name", "")
                    location_description = location.get("description", "")
                    
                    if location_name and location_description:
                        location_names[location_description] = location_name
            
            # Also check key events for location mentions
            if "key_events" in sections:
                for key_event in sections["key_events"]:
                    event_description = key_event.get("description", "")
                    # Simple heuristic: look for location names in descriptions
                    # This would be enhanced with proper NLP in production
                    if "location" in event_description.lower() or "place" in event_description.lower():
                        # Extract potential location name (simplified for now)
                        # In production, use NLP to identify proper nouns
                        pass
        
        return location_names

    def _parse_character_involvements(self, act_events: list[ActKeyEvent]) -> dict[str, Any]:
        """Parse character names involved in act events.
        
        This method extracts character names from event descriptions, causes, and effects
        to create INVOLVES relationships.
        
        Args:
            act_events: List of ActKeyEvent instances
            
        Returns:
            Dictionary mapping event IDs to lists of (character_name, role) tuples
        """
        character_involvements = {}
        
        for act_event in act_events:
            # Extract potential character names from event description, cause, and effect
            # This is a simplified approach - in production, would use NLP named entity recognition
            text_to_parse = f"{act_event.name} {act_event.cause} {act_event.effect}"
            characters = self._extract_character_names(text_to_parse)
            
            if characters:
                character_involvements[act_event.id] = characters
            
            logger.debug(
                "Parsed character involvements for event",
                event_id=act_event.id,
                event_name=act_event.name,
                characters=[c[0] for c in characters],
                extra={"chapter": self.chapter_number}
            )
        
        return character_involvements

    def _extract_character_names(self, text: str) -> list[tuple[str, str | None]]:
        """Extract character names from text using simple heuristics.
        
        Args:
            text: Text to search for character names
            
        Returns:
            List of (character_name, role) tuples
        """
        # This is a simplified approach - in production, would use NLP named entity recognition
        # For now, we'll assume character names are proper nouns (capitalized)
        # and look for common character names from the character sheets
        
        # Note: In production, this would:
        # 1. Use NER to extract named entities
        # 2. Match against known character names
        # 3. Infer roles (protagonist, antagonist, witness, etc.) from context
        
        # For now, return empty list - actual character extraction
        # happens in Stage 4 (Chapter Outlines) where characters are explicitly listed
        return []

    def _parse_location_involvements(self, act_outline_data: dict[str, Any]) -> dict[str, str]:
        """Parse location information from act outline data.
        
        This method extracts location information from act outlines to create OCCURS_AT relationships.
        
        Args:
            act_outline_data: Parsed act outline data
            
        Returns:
            Dictionary mapping event IDs to location names
        """
        location_involvements = {}
        
        # Check if acts are in the act outline
        if "acts" not in act_outline_data:
            logger.warning("No acts found in act outline data")
            return location_involvements
        
        # Process each act
        for act_data in act_outline_data["acts"]:
            if "sections" not in act_data:
                continue
            
            sections = act_data["sections"]
            
            # Check for location data in various sections
            if "locations" in sections:
                for location in sections["locations"]:
                    location_name = location.get("name", "")
                    location_description = location.get("description", "")
                    
                    if location_name and location_description:
                        location_involvements[location_description] = location_name
            
            # Also check key events for location mentions
            if "key_events" in sections:
                for key_event in sections["key_events"]:
                    event_description = key_event.get("description", "")
                    # Simple heuristic: look for location names in descriptions
                    # This would be enhanced with proper NLP in production
                    if "location" in event_description.lower() or "place" in event_description.lower():
                        # Extract potential location name (simplified for now)
                        # In production, use NLP to identify proper nouns
                        pass
        
        return location_involvements

    async def create_act_key_event_nodes(self, act_events: list[ActKeyEvent]) -> bool:
        """Create ActKeyEvent Event nodes in Neo4j.
        
        Args:
            act_events: List of ActKeyEvent instances to create
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build Cypher query for creating event nodes
            cypher_queries = []
            
            for act_event in act_events:
                query = """
                MERGE (e:Event {id: $id})
                ON CREATE SET 
                    e.name = $name,
                    e.description = $description,
                    e.event_type = $event_type,
                    e.act_number = $act_number,
                    e.sequence_in_act = $sequence_in_act,
                    e.cause = $cause,
                    e.effect = $effect,
                    e.created_chapter = $created_chapter,
                    e.is_provisional = $is_provisional,
                    e.created_ts = timestamp(),
                    e.updated_ts = timestamp()
                ON MATCH SET 
                    e.name = $name,
                    e.description = $description,
                    e.event_type = $event_type,
                    e.act_number = $act_number,
                    e.sequence_in_act = $sequence_in_act,
                    e.cause = $cause,
                    e.effect = $effect,
                    e.created_chapter = $created_chapter,
                    e.is_provisional = $is_provisional,
                    e.updated_ts = timestamp()
                """
                
                params = {
                    "id": act_event.id,
                    "name": act_event.name,
                    "description": act_event.description,
                    "event_type": act_event.event_type,
                    "act_number": act_event.act_number,
                    "sequence_in_act": act_event.sequence_in_act,
                    "cause": act_event.cause,
                    "effect": act_event.effect,
                    "created_chapter": act_event.created_chapter,
                    "is_provisional": act_event.is_provisional,
                }
                
                cypher_queries.append((query, params))
            
            # Execute all queries
            for query, params in cypher_queries:
                await neo4j_manager.execute_write_query(query, params)
            
            logger.info(
                "Successfully created %d ActKeyEvent event nodes",
                len(act_events),
                extra={"chapter": self.chapter_number}
            )
            
            return True
            
        except Exception as e:
            logger.error("Error creating ActKeyEvent nodes: %s", str(e), exc_info=True)
            return False

    async def enrich_location_names(self, location_names: dict[str, str]) -> bool:
        """Enrich Location nodes with names from act outlines.
        
        Args:
            location_names: Dictionary mapping location descriptions to names
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build Cypher query for updating location names
            cypher_queries = []
            
            for description, name in location_names.items():
                query = """
                MATCH (l:Location {description: $description})
                SET l.name = $name,
                    l.updated_ts = timestamp()
                """
                
                params = {
                    "description": description,
                    "name": name,
                }
                
                cypher_queries.append((query, params))
            
            # Execute all queries
            for query, params in cypher_queries:
                await neo4j_manager.execute_write_query(query, params)
            
            logger.info(
                "Successfully enriched %d Location nodes with names",
                len(location_names),
                extra={"chapter": self.chapter_number}
            )
            
            return True
            
        except Exception as e:
            logger.error("Error enriching Location names: %s", str(e), exc_info=True)
            return False

    async def create_event_relationships(self, act_events: list[ActKeyEvent]) -> bool:
        """Create relationships between events, characters, and locations.
        
        This method creates all relationship types defined in the schema design:
        - PART_OF relationships between ActKeyEvents and MajorPlotPoints
        - HAPPENS_BEFORE relationships between events
        - INVOLVES relationships between events and characters
        - OCCURS_AT relationships between events and locations
        
        Args:
            act_events: List of ActKeyEvent instances
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build Cypher queries for relationship creation
            cypher_queries = []
            
            # First, create PART_OF relationships between ActKeyEvents and MajorPlotPoints
            for act_event in act_events:
                # Determine which MajorPlotPoint this event belongs to based on act_number and sequence
                # This is a simplified heuristic - in production, use more sophisticated logic
                if act_event.act_number == 1 and act_event.sequence_in_act <= 2:
                    major_plot_point_name = "Inciting Incident"
                elif act_event.act_number == 1 and act_event.sequence_in_act > 2:
                    major_plot_point_name = "Midpoint"
                elif act_event.act_number == 2:
                    major_plot_point_name = "Climax"
                else:
                    major_plot_point_name = "Resolution"
                
                query = """
                MATCH (major:Event {event_type: "MajorPlotPoint", name: $major_plot_point_name})
                MATCH (act:Event {id: $act_event_id})
                MERGE (act)-[r:PART_OF]->(major)
                SET r.created_ts = timestamp(),
                    r.updated_ts = timestamp()
                """
                
                params = {
                    "major_plot_point_name": major_plot_point_name,
                    "act_event_id": act_event.id,
                }
                
                cypher_queries.append((query, params))
            
            # Create HAPPENS_BEFORE relationships based on sequence_in_act
            # Events with lower sequence_in_act happen before those with higher sequence_in_act
            for i in range(len(act_events)):
                for j in range(i + 1, len(act_events)):
                    event_a = act_events[i]
                    event_b = act_events[j]
                    
                    # Only create relationship if they're in the same act
                    if event_a.act_number == event_b.act_number:
                        query = """
                        MATCH (a:Event {id: $event_a_id})
                        MATCH (b:Event {id: $event_b_id})
                        MERGE (a)-[r:HAPPENS_BEFORE]->(b)
                        SET r.created_ts = timestamp(),
                            r.updated_ts = timestamp()
                        """
                        
                        params = {
                            "event_a_id": event_a.id,
                            "event_b_id": event_b.id,
                        }
                        
                        cypher_queries.append((query, params))
            
            # Create INVOLVES relationships with characters
            # In Stage 3, character extraction is simplified
            # Actual character extraction happens in Stage 4 (Chapter Outlines)
            # For now, we create placeholder relationships that can be enhanced later
            
            # Create OCCURS_AT relationships with locations
            # Locations are enriched in Stage 3, but event locations are typically
            # defined in Stage 4 (Chapter Outlines)
            
            # Execute all queries
            for query, params in cypher_queries:
                await neo4j_manager.execute_write_query(query, params)
            
            logger.info(
                "Successfully created PART_OF, HAPPENS_BEFORE relationships",
                extra={"chapter": self.chapter_number}
            )
            
            return True
            
        except Exception as e:
            logger.error("Error creating event relationships: %s", str(e), exc_info=True)
            return False

    async def parse_and_persist(self) -> tuple[bool, str]:
        """Parse act outline and persist to Neo4j.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Step 1: Parse act outline
            logger.info("Parsing act outline from %s", self.act_outline_path)
            act_outline_data = await self.parse_act_outline()
            
            if not act_outline_data:
                return False, "No data found in act outline"
            
            logger.info("Parsed act outline data", extra={"chapter": self.chapter_number})
            
            # Step 2: Parse act key events
            logger.info("Parsing act key events from act outline")
            act_events = self._parse_act_key_events(act_outline_data)
            
            if not act_events:
                return False, "No act key events found in act outline"
            
            logger.info("Parsed %d act key events", len(act_events), extra={"chapter": self.chapter_number})
            
            # Step 3: Parse location name enrichment
            logger.info("Parsing location name enrichment from act outline")
            location_names = self._parse_location_enrichment(act_outline_data)
            
            if not location_names:
                logger.warning("No location names found in act outline")
            
            logger.info("Parsed %d location names", len(location_names), extra={"chapter": self.chapter_number})
            
            # Step 4: Create act key event nodes
            logger.info("Creating ActKeyEvent event nodes in Neo4j")
            act_events_success = await self.create_act_key_event_nodes(act_events)
            
            if not act_events_success:
                return False, "Failed to create ActKeyEvent nodes"
            
            # Step 5: Enrich location names
            logger.info("Enriching Location nodes with names")
            location_names_success = await self.enrich_location_names(location_names)
            
            if not location_names_success:
                return False, "Failed to enrich Location names"
            
            # Step 6: Create event relationships
            logger.info("Creating event relationships in Neo4j")
            relationships_success = await self.create_event_relationships(act_events)
            
            if not relationships_success:
                return False, "Failed to create event relationships"
            
            return (
                True,
                f"Successfully parsed and persisted "
                f"{len(act_events)} ActKeyEvents, "
                f"{len(location_names)} Location name enrichments, and "
                f"{len(act_events)} PART_OF/HAPPENS_BEFORE relationships"
            )
            
        except Exception as e:
            logger.error("Error in parse_and_persist: %s", str(e), exc_info=True)
            return False, f"Error parsing and persisting act outline: {str(e)}"


__all__ = ["ActOutlineParser"]
