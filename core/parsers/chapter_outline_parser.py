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

from core.db_manager import neo4j_manager
from models.kg_models import Chapter, CharacterProfile, Location, Scene, SceneEvent

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
            with open(self.chapter_outline_path, encoding="utf-8") as f:
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
        hash_obj = hashlib.sha256(hash_input.encode("utf-8"))
        hash_hex = hash_obj.hexdigest()[:16]  # Use first 16 chars for brevity
        return f"{entity_type}_{hash_hex}"

    def _parse_chapter(self, chapter_outline_data: dict[str, Any]) -> Chapter:
        """Parse chapter data from chapter outline.

        Args:
            chapter_outline_data: Parsed chapter outline data (single chapter dict)

        Returns:
            Chapter instance
        """
        chapter_number = chapter_outline_data.get("chapter_number", 0)
        act_number = chapter_outline_data.get("act_number", 0)

        chapter_id = self._generate_id("Chapter", chapter_number)

        # Use the summary field directly from chapter outline data
        summary = chapter_outline_data.get("summary", "")

        chapter = Chapter(
            id=chapter_id,
            number=chapter_number,
            title=chapter_outline_data.get("title", f"Chapter {chapter_number}"),
            summary=summary,
            act_number=act_number,
            created_chapter=chapter_number,
            is_provisional=False,
            created_ts=None,
            updated_ts=None,
        )

        return chapter

    def _parse_scenes(self, chapter_outline_data: dict[str, Any], character_names: list[str]) -> list[Scene]:
        """Parse scenes from chapter outline data.

        Args:
            chapter_outline_data: Parsed chapter outline data (single chapter dict)
            character_names: List of known character names from Neo4j

        Returns:
            List of Scene instances
        """
        scenes = []

        chapter_number = chapter_outline_data.get("chapter_number", 0)
        scene_description = chapter_outline_data.get("scene_description", "")
        plot_point = chapter_outline_data.get("plot_point", "")
        key_beats = chapter_outline_data.get("key_beats", [])

        if not scene_description:
            logger.warning("No scene_description found in chapter outline data")
            return scenes

        scene_index = 0
        scene_id = self._generate_id("Scene", chapter_number, scene_index)

        pov_character = self._extract_pov_character_from_beats(key_beats, character_names)

        scene = Scene(
            id=scene_id,
            chapter_number=chapter_number,
            scene_index=scene_index,
            title=f"Chapter {chapter_number} Scene {scene_index + 1}",
            pov_character=pov_character,
            setting=scene_description,
            plot_point=plot_point,
            conflict="",
            outcome="",
            beats=key_beats,
            created_chapter=chapter_number,
            is_provisional=False,
            created_ts=None,
            updated_ts=None,
        )

        scenes.append(scene)

        return scenes

    def _extract_pov_character_from_beats(self, key_beats: list[str], character_names: list[str]) -> str:
        """Extract POV character from key beats using simple heuristic.

        Args:
            key_beats: List of key beat descriptions
            character_names: List of known character names from Neo4j

        Returns:
            POV character name (first character mentioned, or empty string)
        """
        if not key_beats:
            return ""

        # First, try to find POV character in provided character names
        for beat in key_beats:
            for name in character_names:
                if name in beat:
                    return name

        # If no character_names provided, extract the first character mentioned in beats
        # by looking for capitalized names (assuming proper names start with capital letters)
        for beat in key_beats:
            # Split by spaces and look for capitalized words that could be names
            words = beat.split()
            for i, word in enumerate(words):
                # Check if word is a proper noun (starts with capital letter and is not a common article)
                if len(word) > 1 and word[0].isupper() and word[1].islower():
                    # Filter out common words that are not names
                    common_words = {"the", "a", "an", "and", "but", "or", "for", "nor", "so", "yet"}
                    if word.lower() not in common_words:
                        # Try to extend to multi-word name if the next word also starts with capital
                        if i + 1 < len(words) and words[i + 1][0].isupper():
                            full_name = f"{word} {words[i + 1]}"
                            # Validate it's not just a common two-word phrase
                            if full_name.lower() not in {"the end", "the end of", "the beginning", "the start of", "the story begins", "the story ends"}:
                                return full_name
                        return word

        return ""

    def _parse_scene_events(self, chapter_outline_data: dict[str, Any], character_names: list[str]) -> list[SceneEvent]:
        """Parse scene events from chapter outline data.

        Args:
            chapter_outline_data: Parsed chapter outline data (single chapter dict)
            character_names: List of known character names from Neo4j

        Returns:
            List of SceneEvent instances
        """
        events = []

        chapter_number = chapter_outline_data.get("chapter_number", 0)
        act_number = chapter_outline_data.get("act_number", 0)
        key_beats = chapter_outline_data.get("key_beats", [])

        if not key_beats:
            return events

        scene_index = 0

        for event_index, beat in enumerate(key_beats):
            event_id = self._generate_id("Event", chapter_number, scene_index, event_index)

            pov_character = self._extract_pov_character_from_beats([beat], character_names)

            scene_event = SceneEvent(
                id=event_id,
                name=beat,
                description=beat,
                event_type="SceneEvent",
                chapter_number=chapter_number,
                act_number=act_number,
                scene_index=scene_index,
                conflict="",
                outcome="",
                pov_character=pov_character,
                created_chapter=chapter_number,
                is_provisional=False,
                created_ts=None,
                updated_ts=None,
            )

            events.append(scene_event)

        return events

    def _parse_locations(
        self,
        chapter_outline_data: dict[str, Any],
        known_locations: list[dict[str, str]],
        chapter_number: int,
    ) -> list[Location]:
        """Match known locations against chapter outline text.

        Searches `scene_description` and `key_beats` for references to locations
        that already exist in the knowledge graph (from Stage 3 act outlines).

        Args:
            chapter_outline_data: Parsed chapter outline data (single chapter dict).
            known_locations: Location dicts from Neo4j with 'id', 'name', 'description'.
            chapter_number: Current chapter number, used to tag returned locations
                for OCCURS_AT relationship matching.

        Returns:
            List of Location instances representing existing locations referenced
            in this chapter's outline text.
        """
        if not known_locations:
            return []

        scene_description = chapter_outline_data.get("scene_description", "")
        key_beats = chapter_outline_data.get("key_beats", [])
        searchable_text = " ".join([scene_description] + key_beats).lower()

        if not searchable_text.strip():
            return []

        matched: list[Location] = []
        for location_data in known_locations:
            location_name = location_data["name"]
            if location_name.lower() in searchable_text:
                matched.append(
                    Location(
                        id=location_data["id"],
                        name=location_name,
                        description=location_data.get("description", ""),
                        category="Location",
                        created_chapter=chapter_number,
                    )
                )

        return matched

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
                MERGE (c:Chapter {number: $number})
                ON CREATE SET
                    c.id = $id,
                    c.title = $title,
                    c.summary = $summary,
                    c.act_number = $act_number,
                    c.created_chapter = $created_chapter,
                    c.is_provisional = $is_provisional,
                    c.created_ts = timestamp(),
                    c.updated_ts = timestamp()
                ON MATCH SET
                    c.id = $id,
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

            logger.info("Successfully created %d Chapter nodes", len(chapters), extra={"chapter": self.chapter_number})

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

            logger.info("Successfully created %d Scene nodes", len(scenes), extra={"chapter": self.chapter_number})

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
                MERGE (e:Event {name: $name})
                ON CREATE SET
                    e.id = $id,
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
                    e.id = $id,
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

            logger.info("Successfully created %d SceneEvent nodes", len(events), extra={"chapter": self.chapter_number})

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

            logger.info("Successfully created %d Location nodes", len(locations), extra={"chapter": self.chapter_number})

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
                if not scene.pov_character:
                    continue
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
                    if scene.chapter_number == location.created_chapter:
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
                    if event.chapter_number == scene.chapter_number and event.scene_index == scene.scene_index:
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
                if not event.pov_character:
                    continue
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

            logger.info("Successfully created %d relationships", len(cypher_queries), extra={"chapter": self.chapter_number})

            return True

        except Exception as e:
            logger.error("Error creating relationships: %s", str(e), exc_info=True)
            return False

    async def _get_all_character_names(self) -> list[str]:
        """Get all character names from Neo4j.

        Returns:
            List of character names
        """
        try:
            query = "MATCH (c:Character) RETURN c.name as name ORDER BY c.name"
            result = await neo4j_manager.execute_read_query(query, {})
            return [record["name"] for record in result if record.get("name")]
        except Exception as e:
            logger.error("Error fetching character names: %s", str(e), exc_info=True)
            return []

    async def _get_all_locations(self) -> list[dict[str, str]]:
        """Get all location names and IDs from Neo4j.

        Returns:
            List of dicts with 'id', 'name', and 'description' keys.
        """
        query = """
        MATCH (l:Location)
        WHERE l.name IS NOT NULL
        RETURN l.id as id, l.name as name, l.description as description
        ORDER BY l.name
        """
        result = await neo4j_manager.execute_read_query(query, {})
        return [
            {
                "id": record["id"],
                "name": record["name"],
                "description": record.get("description", ""),
            }
            for record in result
            if record.get("name")
        ]

    async def _get_character_by_name(self, character_name: str) -> CharacterProfile | None:
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
        results = await neo4j_manager.execute_read_query(query, {"act_number": act_number, "sequence_in_act": sequence_in_act})
        return results[0] if results else None

    async def parse_and_persist(self) -> tuple[bool, str]:
        """Parse chapter outline and persist to Neo4j.

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            logger.info("Parsing chapter outline from %s", self.chapter_outline_path)
            all_chapter_data = await self.parse_chapter_outline()

            if not all_chapter_data:
                return False, "No data found in chapter outline"

            logger.info("Parsed chapter outline data", extra={"chapter": self.chapter_number})

            logger.info("Fetching character names for POV extraction")
            character_names = await self._get_all_character_names()

            logger.info("Fetching known locations for scene-location matching")
            known_locations = await self._get_all_locations()

            all_chapters = []
            all_scenes = []
            all_events = []
            all_referenced_locations: list[Location] = []

            for chapter_key, chapter_outline_data in all_chapter_data.items():
                if not isinstance(chapter_outline_data, dict):
                    continue

                logger.info("Parsing chapter from chapter outline")
                chapter = self._parse_chapter(chapter_outline_data)

                if not chapter:
                    logger.warning("Could not parse chapter %s", chapter_key)
                    continue

                logger.info("Parsed chapter", extra={"chapter": chapter.number})

                logger.info("Parsing scenes from chapter outline")
                scenes = self._parse_scenes(chapter_outline_data, character_names)

                if not scenes:
                    logger.warning("No scenes found in chapter outline")

                logger.info("Parsed %d scenes", len(scenes), extra={"chapter": chapter.number})

                logger.info("Parsing scene events from chapter outline")
                events = self._parse_scene_events(chapter_outline_data, character_names)

                if not events:
                    logger.warning("No scene events found in chapter outline")

                logger.info("Parsed %d scene events", len(events), extra={"chapter": chapter.number})

                logger.info("Matching locations from chapter outline")
                locations = self._parse_locations(chapter_outline_data, known_locations, chapter.number)

                if not locations:
                    logger.debug(
                        "No location references found in chapter outline",
                        extra={"chapter": chapter.number},
                    )

                logger.info("Matched %d locations", len(locations), extra={"chapter": chapter.number})

                all_chapters.append(chapter)
                all_scenes.extend(scenes)
                all_events.extend(events)
                all_referenced_locations.extend(locations)

            if not all_chapters:
                return False, "No chapters found in chapter outline"

            logger.info("Creating Chapter nodes in Neo4j")
            chapters_success = await self.create_chapter_nodes(all_chapters)

            if not chapters_success:
                return False, "Failed to create Chapter nodes"

            logger.info("Creating Scene nodes in Neo4j")
            scenes_success = await self.create_scene_nodes(all_scenes)

            if not scenes_success:
                return False, "Failed to create Scene nodes"

            logger.info("Creating SceneEvent nodes in Neo4j")
            events_success = await self.create_scene_event_nodes(all_events)

            if not events_success:
                return False, "Failed to create SceneEvent nodes"

            logger.info("Creating relationships in Neo4j")
            relationships_success = await self.create_relationships(
                all_chapters, all_scenes, all_events, all_referenced_locations,
            )

            if not relationships_success:
                return False, "Failed to create relationships"

            return (
                True,
                f"Successfully parsed and persisted "
                f"{len(all_chapters)} Chapter, "
                f"{len(all_scenes)} Scenes, "
                f"{len(all_events)} SceneEvents, "
                f"{len(all_referenced_locations)} location references, and "
                f"{len(all_scenes) + len(all_events) + len(all_referenced_locations)} relationships",
            )

        except Exception as e:
            logger.error("Error in parse_and_persist: %s", str(e), exc_info=True)
            return False, f"Error parsing and persisting chapter outline: {str(e)}"


__all__ = ["ChapterOutlineParser"]
