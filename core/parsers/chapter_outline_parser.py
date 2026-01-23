# core/parsers/chapter_outline_parser.py
"""Parse chapter outlines and create Stage 4 knowledge graph entities.

This module provides the ChapterOutlineParser class that:
1. Reads chapter outlines from JSON files
2. Creates Chapter nodes (1 per chapter)
3. Creates Scene nodes (3-5 per chapter)
4. Creates SceneEvent nodes (1-3 per scene)
5. Creates Location nodes (scene-specific locations)
6. Creates relationships between all these entities
7. Persists to Neo4j using the existing data access layer

Based on: docs/schema-design.md - Stage 4: Chapter Outlines
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import structlog
from pydantic import BaseModel, Field

from core.db_manager import neo4j_manager
from core.exceptions import DatabaseError
from models.kg_models import Chapter, Scene, SceneEvent, Location
from utils.common import try_load_json_from_response

logger = structlog.get_logger(__name__)


class ChapterOutlineParser:
    """Parse chapter outlines and create Stage 4 knowledge graph entities.
    
    This parser handles Stage 4 of the knowledge graph construction:
    - Chapter node creation (1 per chapter)
    - Scene node creation (3-5 per chapter)
    - SceneEvent node creation (1-3 per scene)
    - Location node creation (scene-specific locations)
    - Relationship creation between all entities
    - Persistence to Neo4j
    
    Attributes:
        chapter_outline_path: Path to chapter outline JSON file
        chapter_number: Chapter number for provenance
    """

    def __init__(self, chapter_outline_path: str = "chapter_outlines/chapter_{N}_v1.json", chapter_number: int = 0):
        """Initialize the ChapterOutlineParser.
        
        Args:
            chapter_outline_path: Path to chapter outline JSON file
            chapter_number: Chapter number for provenance
        """
        self.chapter_outline_path = chapter_outline_path
        self.chapter_number = chapter_number

    async def parse_chapter_outline(self) -> dict[str, Any]:
        """Parse chapter outline from JSON file.
        
        Returns:
            Dictionary containing parsed chapter outline data
            
        Raises:
            ValueError: If chapter outline file cannot be read or parsed
            DatabaseError: If there are issues persisting to Neo4j
        """
        try:
            # Read the chapter outline JSON file
            with open(self.chapter_outline_path, 'r', encoding='utf-8') as f:
                chapter_outline_data = json.load(f)
        except FileNotFoundError as e:
            logger.error(f"Chapter outline file not found: {self.chapter_outline_path}", exc_info=True)
            raise ValueError(f"Chapter outline file not found: {self.chapter_outline_path}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in chapter outline file: {self.chapter_outline_path}", exc_info=True)
            raise ValueError(f"Invalid JSON in chapter outline file: {self.chapter_outline_path}") from e

        return chapter_outline_data

    def _generate_id(self, entity_type: str, chapter_number: int, scene_index: int = 0, event_index: int = 0) -> str:
        """Generate a stable ID for an entity.
        
        Args:
            entity_type: Type of entity (Chapter, Scene, Event, Location)
            chapter_number: Chapter number
            scene_index: Scene index (for scenes and events)
            event_index: Event index (for events)
            
        Returns:
            Generated entity ID
        """
        # Use SHA256 hash for stable ID generation
        hash_input = f"{entity_type}_{chapter_number}_{scene_index}_{event_index}"
        hash_obj = hashlib.sha256(hash_input.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()[:16]  # Use first 16 chars for brevity
        return f"{entity_type}_{hash_hex}"

    def _parse_chapter(self, chapter_outline_data: dict[str, Any]) -> Chapter:
        """Parse chapter data from chapter outline.
        
        Args:
            chapter_outline_data: Parsed chapter outline data
            
        Returns:
            Chapter instance
        """
        chapter_number = chapter_outline_data.get("chapter_number", 0)
        act_number = chapter_outline_data.get("act_number", 0)
        
        # Generate chapter ID
        chapter_id = self._generate_id("Chapter", chapter_number)
        
        # Create Chapter
        chapter = Chapter(
            id=chapter_id,
            number=chapter_number,
            title=chapter_outline_data.get("title", f"Chapter {chapter_number}"),
            summary=chapter_outline_data.get("summary", ""),
            act_number=act_number,
            created_chapter=chapter_number,
            is_provisional=False,
            created_ts=None,  # Will be set by Neo4j
            updated_ts=None,  # Will be set by Neo4j
        )
        
        return chapter

    def _parse_scenes(self, chapter_outline_data: dict[str, Any]) -> list[Scene]:
        """Parse scenes from chapter outline data.
        
        Args:
            chapter_outline_data: Parsed chapter outline data
            
        Returns:
            List of Scene instances
        """
        scenes = []
        
        # Check if scenes are in the chapter outline
        if "scenes" not in chapter_outline_data:
            logger.warning("No scenes found in chapter outline data")
            return scenes
        
        # Process each scene in the chapter
        for scene_data in chapter_outline_data["scenes"]:
            scene_index = scene_data.get("scene_index", 0)
            chapter_number = chapter_outline_data.get("chapter_number", 0)
            
            # Generate scene ID
            scene_id = self._generate_id("Scene", chapter_number, scene_index)
            
            # Create Scene
            scene = Scene(
                id=scene_id,
                chapter_number=chapter_number,
                scene_index=scene_index,
                title=scene_data.get("title", f"Scene {scene_index + 1}"),
                pov_character=scene_data.get("pov_character", ""),
                setting=scene_data.get("setting", ""),
                plot_point=scene_data.get("plot_point", ""),
                conflict=scene_data.get("conflict", ""),
                outcome=scene_data.get("outcome", ""),
                beats=scene_data.get("beats", []),
                created_chapter=chapter_number,
                is_provisional=False,
                created_ts=None,  # Will be set by Neo4j
                updated_ts=None,  # Will be set by Neo4j
            )
            
            scenes.append(scene)
        
        return scenes

    def _parse_scene_events(self, chapter_outline_data: dict[str, Any]) -> list[SceneEvent]:
        """Parse scene events from chapter outline data.
        
        Args:
            chapter_outline_data: Parsed chapter outline data
            
        Returns:
            List of SceneEvent instances
        """
        events = []
        
        # Check if scenes are in the chapter outline
        if "scenes" not in chapter_outline_data:
            logger.warning("No scenes found in chapter outline data")
            return events
        
        # Process each scene for events
        for scene_data in chapter_outline_data["scenes"]:
            scene_index = scene_data.get("scene_index", 0)
            chapter_number = chapter_outline_data.get("chapter_number", 0)
            act_number = chapter_outline_data.get("act_number", 0)
            
            # Check if events are in the scene
            if "events" not in scene_data:
                continue
            
            # Process each event in the scene
            for event_index, event_data in enumerate(scene_data["events"]):
                # Generate event ID
                event_id = self._generate_id("Event", chapter_number, scene_index, event_index)
                
                # Create SceneEvent
                scene_event = SceneEvent(
                    id=event_id,
                    name=event_data.get("name", f"Event {event_index + 1}"),
                    description=event_data.get("description", ""),
                    event_type="SceneEvent",
                    chapter_number=chapter_number,
                    act_number=act_number,
                    scene_index=scene_index,
                    conflict=event_data.get("conflict", ""),
                    outcome=event_data.get("outcome", ""),
                    pov_character=event_data.get("pov_character", ""),
                    created_chapter=chapter_number,
                    is_provisional=False,
                    created_ts=None,  # Will be set by Neo4j
                    updated_ts=None,  # Will be set by Neo4j
                )
                
                events.append(scene_event)
        
        return events

    def _parse_locations(self, chapter_outline_data: dict[str, Any]) -> list[Location]:
        """Parse locations from chapter outline data.
        
        Args:
            chapter_outline_data: Parsed chapter outline data
            
        Returns:
            List of Location instances
        """
        locations = []
        
        # Check if scenes are in the chapter outline
        if "scenes" not in chapter_outline_data:
            logger.warning("No scenes found in chapter outline data")
            return locations
        
        # Process each scene for locations
        for scene_data in chapter_outline_data["scenes"]:
            scene_index = scene_data.get("scene_index", 0)
            chapter_number = chapter_outline_data.get("chapter_number", 0)
            
            # Check if location is in the scene
            if "location" not in scene_data:
                continue
            
            location_data = scene_data["location"]
            
            # Generate location ID
            location_id = self._generate_id("Location", chapter_number, scene_index)
            
            # Create Location
            location = Location(
                id=location_id,
                name=location_data.get("name", ""),
                description=location_data.get("description", ""),
                category="Location",
                created_chapter=chapter_number,
                is_provisional=False,
                created_ts=None,  # Will be set by Neo4j
                updated_ts=None,  # Will be set by Neo4j
            )
            
            locations.append(location)
        
        return locations

    async def create_chapter_nodes(self, chapters: list[Chapter]) -> bool:
        """Create Chapter nodes in Neo4j.
        
        Args:
            chapters: List of Chapter instances to create
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build Cypher query for creating chapter nodes
            cypher_queries = []
            
            for chapter in chapters:
                query = """
                MERGE (c:Chapter {id: $id})
                ON CREATE SET 
                    c.number = $number,
                    c.title = $title,
                    c.summary = $summary,
                    c.act_number = $act_number,
                    c.created_chapter = $created_chapter,
                    c.is_provisional = $is_provisional,
                    c.created_ts = timestamp(),
                    c.updated_ts = timestamp()
                ON MATCH SET 
                    c.number = $number,
                    c.title = $title,
                    c.summary = $summary,
                    c.act_number = $act_number,
                    c.created_chapter = $created_chapter,
                    c.is_provisional = $is_provisional,
                    c.updated_ts = timestamp()
                """
                
                params = {
                    "id": chapter.id,
                    "number": chapter.number,
                    "title": chapter.title,
                    "summary": chapter.summary,
                    "act_number": chapter.act_number,
                    "created_chapter": chapter.created_chapter,
                    "is_provisional": chapter.is_provisional,
                }
                
                cypher_queries.append((query, params))
            
            # Execute all queries
            for query, params in cypher_queries:
                await neo4j_manager.execute_write_query(query, params)
            
            logger.info(
                "Successfully created %d Chapter nodes",
                len(chapters),
                extra={"chapter": self.chapter_number}
            )
            
            return True
            
        except Exception as e:
            logger.error("Error creating Chapter nodes: %s", str(e), exc_info=True)
            return False

    async def create_scene_nodes(self, scenes: list[Scene]) -> bool:
        """Create Scene nodes in Neo4j.
        
        Args:
            scenes: List of Scene instances to create
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build Cypher query for creating scene nodes
            cypher_queries = []
            
            for scene in scenes:
                query = """
                MERGE (s:Scene {id: $id})
                ON CREATE SET 
                    s.chapter_number = $chapter_number,
                    s.scene_index = $scene_index,
                    s.title = $title,
                    s.pov_character = $pov_character,
                    s.setting = $setting,
                    s.plot_point = $plot_point,
                    s.conflict = $conflict,
                    s.outcome = $outcome,
                    s.beats = $beats,
                    s.created_chapter = $created_chapter,
                    s.is_provisional = $is_provisional,
                    s.created_ts = timestamp(),
                    s.updated_ts = timestamp()
                ON MATCH SET 
                    s.chapter_number = $chapter_number,
                    s.scene_index = $scene_index,
                    s.title = $title,
                    s.pov_character = $pov_character,
                    s.setting = $setting,
                    s.plot_point = $plot_point,
                    s.conflict = $conflict,
                    s.outcome = $outcome,
                    s.beats = $beats,
                    s.created_chapter = $created_chapter,
                    s.is_provisional = $is_provisional,
                    s.updated_ts = timestamp()
                """
                
                params = {
                    "id": scene.id,
                    "chapter_number": scene.chapter_number,
                    "scene_index": scene.scene_index,
                    "title": scene.title,
                    "pov_character": scene.pov_character,
                    "setting": scene.setting,
                    "plot_point": scene.plot_point,
                    "conflict": scene.conflict,
                    "outcome": scene.outcome,
                    "beats": scene.beats,
                    "created_chapter": scene.created_chapter,
                    "is_provisional": scene.is_provisional,
                }
                
                cypher_queries.append((query, params))
            
            # Execute all queries
            for query, params in cypher_queries:
                await neo4j_manager.execute_write_query(query, params)
            
            logger.info(
                "Successfully created %d Scene nodes",
                len(scenes),
                extra={"chapter": self.chapter_number}
            )
            
            return True
            
        except Exception as e:
            logger.error("Error creating Scene nodes: %s", str(e), exc_info=True)
            return False

    async def create_scene_event_nodes(self, events: list[SceneEvent]) -> bool:
        """Create SceneEvent nodes in Neo4j.
        
        Args:
            events: List of SceneEvent instances to create
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build Cypher query for creating event nodes
            cypher_queries = []
            
            for event in events:
                query = """
                MERGE (e:Event {id: $id})
                ON CREATE SET 
                    e.name = $name,
                    e.description = $description,
                    e.event_type = $event_type,
                    e.chapter_number = $chapter_number,
                    e.act_number = $act_number,
                    e.scene_index = $scene_index,
                    e.conflict = $conflict,
                    e.outcome = $outcome,
                    e.pov_character = $pov_character,
                    e.created_chapter = $created_chapter,
                    e.is_provisional = $is_provisional,
                    e.created_ts = timestamp(),
                    e.updated_ts = timestamp()
                ON MATCH SET 
                    e.name = $name,
                    e.description = $description,
                    e.event_type = $event_type,
                    e.chapter_number = $chapter_number,
                    e.act_number = $act_number,
                    e.scene_index = $scene_index,
                    e.conflict = $conflict,
                    e.outcome = $outcome,
                    e.pov_character = $pov_character,
                    e.created_chapter = $created_chapter,
                    e.is_provisional = $is_provisional,
                    e.updated_ts = timestamp()
                """
                
                params = {
                    "id": event.id,
                    "name": event.name,
                    "description": event.description,
                    "event_type": event.event_type,
                    "chapter_number": event.chapter_number,
                    "act_number": event.act_number,
                    "scene_index": event.scene_index,
                    "conflict": event.conflict,
                    "outcome": event.outcome,
                    "pov_character": event.pov_character,
                    "created_chapter": event.created_chapter,
                    "is_provisional": event.is_provisional,
                }
                
                cypher_queries.append((query, params))
            
            # Execute all queries
            for query, params in cypher_queries:
                await neo4j_manager.execute_write_query(query, params)
            
            logger.info(
                "Successfully created %d SceneEvent nodes",
                len(events),
                extra={"chapter": self.chapter_number}
            )
            
            return True
            
        except Exception as e:
            logger.error("Error creating SceneEvent nodes: %s", str(e), exc_info=True)
            return False

    async def create_location_nodes(self, locations: list[Location]) -> bool:
        """Create Location nodes in Neo4j.
        
        Args:
            locations: List of Location instances to create
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build Cypher query for creating location nodes
            cypher_queries = []
            
            for location in locations:
                query = """
                MERGE (l:Location {id: $id})
                ON CREATE SET 
                    l.name = $name,
                    l.description = $description,
                    l.category = $category,
                    l.created_chapter = $created_chapter,
                    l.is_provisional = $is_provisional,
                    l.created_ts = timestamp(),
                    l.updated_ts = timestamp()
                ON MATCH SET 
                    l.name = $name,
                    l.description = $description,
                    l.category = $category,
                    l.created_chapter = $created_chapter,
                    l.is_provisional = $is_provisional,
                    l.updated_ts = timestamp()
                """
                
                params = {
                    "id": location.id,
                    "name": location.name,
                    "description": location.description,
                    "category": location.category,
                    "created_chapter": location.created_chapter,
                    "is_provisional": location.is_provisional,
                }
                
                cypher_queries.append((query, params))
            
            # Execute all queries
            for query, params in cypher_queries:
                await neo4j_manager.execute_write_query(query, params)
            
            logger.info(
                "Successfully created %d Location nodes",
                len(locations),
                extra={"chapter": self.chapter_number}
            )
            
            return True
            
        except Exception as e:
            logger.error("Error creating Location nodes: %s", str(e), exc_info=True)
            return False

    async def create_relationships(self, chapters: list[Chapter], scenes: list[Scene], events: list[SceneEvent], locations: list[Location]) -> bool:
        """Create relationships between all entities.
        
        Args:
            chapters: List of Chapter instances
            scenes: List of Scene instances
            events: List of SceneEvent instances
            locations: List of Location instances
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build Cypher queries for relationship creation
            cypher_queries = []
            
            # Create Scene -[PART_OF]-> Chapter relationships
            for scene in scenes:
                for chapter in chapters:
                    if scene.chapter_number == chapter.number:
                        query = """
                        MATCH (s:Scene {id: $scene_id})
                        MATCH (c:Chapter {id: $chapter_id})
                        MERGE (s)-[r:PART_OF]->(c)
                        SET r.created_ts = timestamp(),
                            r.updated_ts = timestamp()
                        """
                        
                        params = {
                            "scene_id": scene.id,
                            "chapter_id": chapter.id,
                        }
                        
                        cypher_queries.append((query, params))
            
            # Create Scene -[FOLLOWS]-> Scene relationships
            # Sort scenes by scene_index to create proper sequence
            sorted_scenes = sorted(scenes, key=lambda s: s.scene_index)
            for i in range(len(sorted_scenes) - 1):
                current_scene = sorted_scenes[i]
                next_scene = sorted_scenes[i + 1]
                
                if current_scene.chapter_number == next_scene.chapter_number:
                    query = """
                    MATCH (current:Scene {id: $current_id})
                    MATCH (next:Scene {id: $next_id})
                    MERGE (current)-[r:FOLLOWS]->(next)
                    SET r.created_ts = timestamp(),
                        r.updated_ts = timestamp()
                    """
                    
                    params = {
                        "current_id": current_scene.id,
                        "next_id": next_scene.id,
                    }
                    
                    cypher_queries.append((query, params))
            
            # Create Scene -[FEATURES_CHARACTER]-> Character relationships
            for scene in scenes:
                # Check if pov_character exists in Neo4j
                character = await self._get_character_by_name(scene.pov_character)
                if character:
                    query = """
                    MATCH (s:Scene {id: $scene_id})
                    MATCH (c:Character {name: $character_name})
                    MERGE (s)-[r:FEATURES_CHARACTER]->(c)
                    SET r.is_pov = true,
                        r.created_ts = timestamp(),
                        r.updated_ts = timestamp()
                    """
                    params = {
                        "scene_id": scene.id,
                        "character_name": scene.pov_character,
                    }
                    cypher_queries.append((query, params))
            
            # Create Scene -[OCCURS_AT]-> Location relationships
            for scene in scenes:
                for location in locations:
                    # Simple heuristic: match scenes to locations based on chapter and scene index
                    if scene.chapter_number == location.created_chapter and scene.scene_index == 0:
                        query = """
                        MATCH (s:Scene {id: $scene_id})
                        MATCH (l:Location {id: $location_id})
                        MERGE (s)-[r:OCCURS_AT]->(l)
                        SET r.created_ts = timestamp(),
                            r.updated_ts = timestamp()
                        """
                        
                        params = {
                            "scene_id": scene.id,
                            "location_id": location.id,
                        }
                        
                        cypher_queries.append((query, params))
            
            # Create SceneEvent -[OCCURS_IN_SCENE]-> Scene relationships
            for event in events:
                for scene in scenes:
                    if (event.chapter_number == scene.chapter_number and 
                        event.scene_index == scene.scene_index):
                        query = """
                        MATCH (e:Event {id: $event_id})
                        MATCH (s:Scene {id: $scene_id})
                        MERGE (e)-[r:OCCURS_IN_SCENE]->(s)
                        SET r.created_ts = timestamp(),
                            r.updated_ts = timestamp()
                        """
                        
                        params = {
                            "event_id": event.id,
                            "scene_id": scene.id,
                        }
                        
                        cypher_queries.append((query, params))
            
            # Create Event -[INVOLVES]-> Character relationships
            for event in events:
                # Check if pov_character exists in Neo4j
                character = await self._get_character_by_name(event.pov_character)
                if character:
                    query = """
                    MATCH (e:Event {id: $event_id})
                    MATCH (c:Character {name: $character_name})
                    MERGE (e)-[r:INVOLVES]->(c)
                    SET r.role = "protagonist",
                        r.created_ts = timestamp(),
                        r.updated_ts = timestamp()
                    """
                    params = {
                        "event_id": event.id,
                        "character_name": event.pov_character,
                    }
                    cypher_queries.append((query, params))
            
            # Create SceneEvent -[PART_OF]-> ActKeyEvent relationships
            for event in events:
                # Lookup ActKeyEvent for this act and scene
                act_key_event = await self._get_act_key_event(event.act_number, event.scene_index)
                
                if act_key_event:
                    query = """
                    MATCH (e:Event {id: $event_id})
                    MATCH (ake:Event {id: $ake_id})
                    MERGE (e)-[r:PART_OF]->(ake)
                    SET r.created_ts = timestamp(),
                        r.updated_ts = timestamp()
                    """
                    params = {
                        "event_id": event.id,
                        "ake_id": act_key_event.get("id"),
                    }
                    cypher_queries.append((query, params))
            
            # Execute all queries
            for query, params in cypher_queries:
                await neo4j_manager.execute_write_query(query, params)
            
            logger.info(
                "Successfully created %d relationships",
                len(cypher_queries),
                extra={"chapter": self.chapter_number}
            )
            
            return True
            
        except Exception as e:
            logger.error("Error creating relationships: %s", str(e), exc_info=True)
            return False

    async def _get_character_by_name(self, character_name: str) -> "CharacterProfile | None":
        """Query Neo4j for a character by name.
        
        Args:
            character_name: Name of character to look up
            
        Returns:
            CharacterProfile if found, None otherwise
        """
        from data_access.character_queries import get_character_profile_by_name
        return await get_character_profile_by_name(character_name)

    async def _get_act_key_event(self, act_number: int, sequence_in_act: int) -> dict | None:
        """Query Neo4j for an ActKeyEvent by act_number and sequence_in_act.
        
        Args:
            act_number: Act number (1, 2, or 3)
            sequence_in_act: Position within act
            
        Returns:
            Event dict if found, None otherwise
        """
        query = """
        MATCH (e:Event {event_type: "ActKeyEvent", act_number: $act_number, sequence_in_act: $sequence_in_act})
        RETURN e
        """
        results = await neo4j_manager.execute_read_query(query, {
            "act_number": act_number,
            "sequence_in_act": sequence_in_act
        })
        return results[0] if results else None

    async def parse_and_persist(self) -> tuple[bool, str]:
        """Parse chapter outline and persist to Neo4j.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Step 1: Parse chapter outline
            logger.info("Parsing chapter outline from %s", self.chapter_outline_path)
            chapter_outline_data = await self.parse_chapter_outline()
            
            if not chapter_outline_data:
                return False, "No data found in chapter outline"
            
            logger.info("Parsed chapter outline data", extra={"chapter": self.chapter_number})
            
            # Step 2: Parse chapter
            logger.info("Parsing chapter from chapter outline")
            chapter = self._parse_chapter(chapter_outline_data)
            
            if not chapter:
                return False, "No chapter found in chapter outline"
            
            logger.info("Parsed chapter", extra={"chapter": self.chapter_number})
            
            # Step 3: Parse scenes
            logger.info("Parsing scenes from chapter outline")
            scenes = self._parse_scenes(chapter_outline_data)
            
            if not scenes:
                logger.warning("No scenes found in chapter outline")
            
            logger.info("Parsed %d scenes", len(scenes), extra={"chapter": self.chapter_number})
            
            # Step 4: Parse scene events
            logger.info("Parsing scene events from chapter outline")
            events = self._parse_scene_events(chapter_outline_data)
            
            if not events:
                logger.warning("No scene events found in chapter outline")
            
            logger.info("Parsed %d scene events", len(events), extra={"chapter": self.chapter_number})
            
            # Step 5: Parse locations
            logger.info("Parsing locations from chapter outline")
            locations = self._parse_locations(chapter_outline_data)
            
            if not locations:
                logger.warning("No locations found in chapter outline")
            
            logger.info("Parsed %d locations", len(locations), extra={"chapter": self.chapter_number})
            
            # Step 6: Create chapter nodes
            logger.info("Creating Chapter nodes in Neo4j")
            chapters_success = await self.create_chapter_nodes([chapter])
            
            if not chapters_success:
                return False, "Failed to create Chapter nodes"
            
            # Step 7: Create scene nodes
            logger.info("Creating Scene nodes in Neo4j")
            scenes_success = await self.create_scene_nodes(scenes)
            
            if not scenes_success:
                return False, "Failed to create Scene nodes"
            
            # Step 8: Create scene event nodes
            logger.info("Creating SceneEvent nodes in Neo4j")
            events_success = await self.create_scene_event_nodes(events)
            
            if not events_success:
                return False, "Failed to create SceneEvent nodes"
            
            # Step 9: Create location nodes
            logger.info("Creating Location nodes in Neo4j")
            locations_success = await self.create_location_nodes(locations)
            
            if not locations_success:
                return False, "Failed to create Location nodes"
            
            # Step 10: Create relationships
            logger.info("Creating relationships in Neo4j")
            relationships_success = await self.create_relationships([chapter], scenes, events, locations)
            
            if not relationships_success:
                return False, "Failed to create relationships"
            
            return (
                True,
                f"Successfully parsed and persisted "
                f"1 Chapter, "
                f"{len(scenes)} Scenes, "
                f"{len(events)} SceneEvents, "
                f"{len(locations)} Locations, and "
                f"{len(scenes) + len(events) + len(locations)} relationships"
            )
            
        except Exception as e:
            logger.error("Error in parse_and_persist: %s", str(e), exc_info=True)
            return False, f"Error parsing and persisting chapter outline: {str(e)}"


__all__ = ["ChapterOutlineParser"]
