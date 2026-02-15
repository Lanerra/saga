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

import config
from core.db_manager import neo4j_manager
from core.llm_interface_refactored import llm_service
from models.kg_models import ActKeyEvent, Location, WorldItem
from prompts.prompt_renderer import get_system_prompt, render_prompt
from utils.common import try_load_json_from_response
from utils.text_processing import generate_entity_id

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
            with open(self.act_outline_path, encoding="utf-8") as f:
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
        hash_obj = hashlib.sha256(hash_input.encode("utf-8"))
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
        act_key_events: list[ActKeyEvent] = []

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
        location_names: dict[str, str] = {}

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
                        logger.debug(
                            "_parse_location_enrichment: potential location mention in key event",
                            event_description=event_description,
                        )
                        # Note: Full NLP-based extraction would require spaCy or similar
                        # For now, we log the potential location mention for manual review

        return location_names

    async def _parse_character_involvements(self, act_events: list[ActKeyEvent]) -> dict[str, list[tuple[str, str | None]]]:
        """Parse character names involved in act events.

        This method extracts character names from event descriptions, causes, and effects
        to create INVOLVES relationships.

        Args:
            act_events: List of ActKeyEvent instances

        Returns:
            Dictionary mapping event IDs to lists of (character_name, role) tuples
        """
        character_involvements = {}

        known_characters = await self._get_all_character_names()

        for act_event in act_events:
            characters = await self._extract_character_names(act_event.name, act_event.name, act_event.cause, act_event.effect, known_characters)

            if characters:
                character_involvements[act_event.id] = characters
                logger.debug(
                    "Extracted character involvements for event", event_id=act_event.id, event_name=act_event.name, characters=[c[0] for c in characters], extra={"chapter": self.chapter_number}
                )

        logger.info("Parsed character involvements for %d events", len(character_involvements), extra={"chapter": self.chapter_number})

        return character_involvements

    async def _extract_character_names(self, event_name: str, event_description: str, event_cause: str, event_effect: str, known_characters: list[str]) -> list[tuple[str, str | None]]:
        """Extract character names from event text using LLM.

        Args:
            event_name: Event name
            event_description: Event description
            event_cause: What triggers this event
            event_effect: What results from this event
            known_characters: List of known character names

        Returns:
            List of (character_name, role) tuples
        """
        if not known_characters:
            return []

        prompt = render_prompt(
            "knowledge_agent/extract_event_characters.j2",
            {
                "event_name": event_name,
                "event_description": event_description,
                "event_cause": event_cause,
                "event_effect": event_effect,
                "known_characters": known_characters,
            },
        )

        try:
            response, _ = await llm_service.async_call_llm(
                model_name=config.NARRATIVE_MODEL,
                prompt=prompt,
                temperature=0.3,
                max_tokens=8192,
                allow_fallback=True,
                auto_clean_response=True,
                system_prompt=get_system_prompt("knowledge_agent"),
            )

            data, _, parse_errors = try_load_json_from_response(response)

            if data is None:
                logger.warning("Failed to parse character extraction response: %s", parse_errors)
                return []

            if not isinstance(data, list):
                logger.warning("Character extraction response must be array")
                return []

            results = []
            for item in data:
                if isinstance(item, dict) and "name" in item:
                    name = item["name"]
                    role = item.get("role")
                    if name in known_characters:
                        results.append((name, role))

            return results

        except Exception as e:
            logger.warning("Error extracting characters from event: %s", str(e), exc_info=True)
            return []

    async def _parse_location_involvements(self, act_events: list[ActKeyEvent]) -> dict[str, str]:
        """Parse location information from act events.

        This method extracts location information from act events to create OCCURS_AT relationships.

        Args:
            act_events: List of ActKeyEvent instances

        Returns:
            Dictionary mapping event IDs to location names
        """
        location_involvements = {}

        known_locations = await self._get_all_locations()

        for act_event in act_events:
            location = await self._extract_event_location(act_event.name, act_event.name, act_event.cause, act_event.effect, known_locations)

            if location:
                location_involvements[act_event.id] = location
                logger.debug("Extracted location for event", event_id=act_event.id, event_name=act_event.name, location=location, extra={"chapter": self.chapter_number})

        logger.info("Parsed location involvements for %d events", len(location_involvements), extra={"chapter": self.chapter_number})

        return location_involvements

    async def _extract_event_location(self, event_name: str, event_description: str, event_cause: str, event_effect: str, known_locations: list[dict[str, str]]) -> str | None:
        """Extract location where event occurs using LLM.

        Args:
            event_name: Event name
            event_description: Event description
            event_cause: What triggers this event
            event_effect: What results from this event
            known_locations: List of known locations with names and descriptions

        Returns:
            Location name or None
        """
        if not known_locations:
            return None

        prompt = render_prompt(
            "knowledge_agent/extract_event_location.j2",
            {
                "event_name": event_name,
                "event_description": event_description,
                "event_cause": event_cause,
                "event_effect": event_effect,
                "known_locations": known_locations,
            },
        )

        try:
            response, _ = await llm_service.async_call_llm(
                model_name=config.NARRATIVE_MODEL,
                prompt=prompt,
                temperature=0.3,
                max_tokens=8192,
                allow_fallback=True,
                auto_clean_response=True,
                system_prompt=get_system_prompt("knowledge_agent"),
            )

            data, _, parse_errors = try_load_json_from_response(response)

            if data is None:
                logger.warning("Failed to parse location extraction response: %s", parse_errors)
                return None

            if not isinstance(data, dict) or "location" not in data:
                logger.warning("Location extraction response must be object with 'location' key")
                return None

            location = data["location"]
            if location and isinstance(location, str):
                return location

            return None

        except Exception as e:
            logger.warning("Error extracting location from event: %s", str(e), exc_info=True)
            return None

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
        """Get all locations from Neo4j.

        Returns:
            List of dicts with 'name' and 'description' keys
        """
        try:
            query = "MATCH (l:Location) RETURN l.name as name, l.description as description ORDER BY l.name"
            result = await neo4j_manager.execute_read_query(query, {})
            return [{"name": record["name"], "description": record["description"]} for record in result if record.get("name") and record.get("description")]
        except Exception as e:
            logger.error("Error fetching locations: %s", str(e), exc_info=True)
            return []

    async def _get_all_items(self) -> list[dict[str, str]]:
        """Get all items from Neo4j.

        Returns:
            List of dicts with 'name' and 'description' keys
        """
        try:
            query = "MATCH (i:Item) RETURN i.name as name, i.description as description ORDER BY i.name"
            result = await neo4j_manager.execute_read_query(query, {})
            return [{"name": record["name"], "description": record["description"]} for record in result if record.get("name")]
        except Exception as e:
            logger.error("Error fetching items: %s", str(e), exc_info=True)
            return []

    def _collect_act_narrative_text(self, act_outline_data: dict[str, Any]) -> str:
        """Collect all narrative text from act outline for entity extraction.

        Args:
            act_outline_data: Parsed act outline data.

        Returns:
            Concatenated narrative text from all acts.
        """
        text_parts = []
        if "acts" not in act_outline_data:
            return ""

        for act_data in act_outline_data["acts"]:
            if "act_summary" in act_data:
                text_parts.append(act_data["act_summary"])
            sections = act_data.get("sections", {})
            if "opening_situation" in sections:
                text_parts.append(sections["opening_situation"])
            if "act_ending_turn" in sections:
                text_parts.append(sections["act_ending_turn"])
            for key_event in sections.get("key_events", []):
                for field in ("event", "description", "cause", "effect"):
                    value = key_event.get(field, "")
                    if value:
                        text_parts.append(value)
            if "character_development" in sections:
                text_parts.append(str(sections["character_development"]))
            if "stakes_tension" in sections:
                text_parts.append(str(sections["stakes_tension"]))
            if "thematic_threads" in sections:
                text_parts.append(str(sections["thematic_threads"]))

        return "\n".join(text_parts)

    async def extract_new_world_items(self, act_outline_data: dict[str, Any]) -> tuple[list[Location], list[WorldItem]]:
        """Extract new locations and items from act outline text that don't already exist.

        Uses the world items extraction prompt to find entities in the act
        narrative text, then deduplicates against existing Neo4j nodes.

        Args:
            act_outline_data: Parsed act outline data.

        Returns:
            Tuple of (new_locations, new_items) to be created.
        """
        narrative_text = self._collect_act_narrative_text(act_outline_data)
        if not narrative_text:
            return [], []

        prompt = render_prompt(
            "knowledge_agent/extract_world_items_lines.j2",
            {
                "setting": "",
                "outline_text": narrative_text,
            },
        )

        extracted_items: list[dict[str, str]] = []
        for attempt in range(1, config.JSON_PARSE_RETRY_ATTEMPTS + 1):
            try:
                response, _ = await llm_service.async_call_llm(
                    model_name=config.NARRATIVE_MODEL,
                    prompt=prompt,
                    temperature=0.5,
                    max_tokens=config.MAX_GENERATION_TOKENS,
                    allow_fallback=True,
                    auto_clean_response=True,
                    system_prompt=get_system_prompt("knowledge_agent"),
                )

                data, _, _ = try_load_json_from_response(response)
                if data and isinstance(data, list):
                    extracted_items = [item for item in data if isinstance(item, dict) and all(k in item for k in ("name", "category", "description"))]
                    break
            except (json.JSONDecodeError, ValueError) as parse_error:
                if attempt == config.JSON_PARSE_RETRY_ATTEMPTS:
                    logger.warning("Failed to extract world items from act outline: %s", str(parse_error))
                    return [], []

        if not extracted_items:
            return [], []

        existing_locations = await self._get_all_locations()
        existing_items = await self._get_all_items()
        existing_location_names = {loc["name"].lower() for loc in existing_locations if loc.get("name")}
        existing_item_names = {item["name"].lower() for item in existing_items if item.get("name")}

        new_locations: list[Location] = []
        new_items: list[WorldItem] = []

        for item in extracted_items:
            name = str(item["name"]).strip()
            category = str(item["category"]).strip()
            description = str(item["description"]).strip()

            if not name or not description:
                continue

            if category == "location" and name.lower() not in existing_location_names:
                new_locations.append(
                    Location(
                        id=generate_entity_id(name, "location"),
                        name=name,
                        description=description,
                        category="Location",
                        created_chapter=0,
                        is_provisional=False,
                    )
                )
                existing_location_names.add(name.lower())

            elif category == "object" and name.lower() not in existing_item_names:
                new_items.append(
                    WorldItem(
                        id=generate_entity_id(name, "object"),
                        name=name,
                        description=description,
                        category="object",
                        created_chapter=0,
                        is_provisional=False,
                    )
                )
                existing_item_names.add(name.lower())

        logger.info(
            "extract_new_world_items: found new entities from act outlines",
            new_locations=len(new_locations),
            new_items=len(new_items),
        )

        return new_locations, new_items

    async def create_new_location_nodes(self, locations: list[Location]) -> bool:
        """Create new Location nodes in Neo4j.

        Args:
            locations: List of Location instances to create.

        Returns:
            True if successful, False otherwise.
        """
        if not locations:
            return True

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

        for query, params in cypher_queries:
            await neo4j_manager.execute_write_query(query, params)

        logger.info("Created %d new Location nodes from act outlines", len(locations))
        return True

    async def create_new_item_nodes(self, items: list[WorldItem]) -> bool:
        """Create new Item nodes in Neo4j.

        Args:
            items: List of WorldItem instances to create.

        Returns:
            True if successful, False otherwise.
        """
        if not items:
            return True

        cypher_queries = []
        for item in items:
            query = """
            MERGE (i:Item {id: $id})
            ON CREATE SET
                i.name = $name,
                i.description = $description,
                i.category = $category,
                i.created_chapter = $created_chapter,
                i.is_provisional = $is_provisional,
                i.created_ts = timestamp(),
                i.updated_ts = timestamp()
            """
            params = {
                "id": item.id,
                "name": item.name,
                "description": item.description,
                "category": item.category,
                "created_chapter": item.created_chapter,
                "is_provisional": item.is_provisional,
            }
            cypher_queries.append((query, params))

        for query, params in cypher_queries:
            await neo4j_manager.execute_write_query(query, params)

        logger.info("Created %d new Item nodes from act outlines", len(items))
        return True

    async def _extract_event_item_involvements(self, act_events: list[ActKeyEvent], items: list[dict[str, str]]) -> dict[str, list[tuple[str, str]]]:
        """Extract which items are featured in which events using LLM.

        Args:
            act_events: List of ActKeyEvent instances
            items: List of dicts with item 'name' and 'description'

        Returns:
            Dictionary mapping event IDs to list of (item_name, role) tuples
        """
        item_involvements = {}

        for event in act_events:
            prompt = render_prompt(
                "knowledge_agent/extract_event_item.j2",
                {
                    "event_name": event.name,
                    "event_description": event.description,
                    "event_cause": event.cause,
                    "event_effect": event.effect,
                    "known_items": items,
                },
            )

            for attempt in range(1, config.JSON_PARSE_RETRY_ATTEMPTS + 1):
                try:
                    response, _ = await llm_service.async_call_llm(
                        model_name=config.NARRATIVE_MODEL,
                        prompt=prompt,
                        temperature=0.3,
                        max_tokens=config.MAX_GENERATION_TOKENS,
                        allow_fallback=True,
                        auto_clean_response=True,
                        system_prompt=get_system_prompt("knowledge_agent"),
                    )

                    data, _, _ = try_load_json_from_response(response)
                    if not data or not isinstance(data, dict) or "featured_items" not in data:
                        if attempt == 2:
                            break
                        continue

                    featured_items = []
                    for item_data in data["featured_items"]:
                        item_name = item_data.get("item", "")
                        role = item_data.get("role", "featured")
                        if item_name:
                            featured_items.append((item_name, role))

                    if featured_items:
                        item_involvements[event.id] = featured_items

                    break

                except (json.JSONDecodeError, ValueError) as e:
                    if attempt == 2:
                        logger.warning("Failed to extract item involvement for event %s after %d attempts: %s", event.id, attempt, str(e), exc_info=True)

        logger.info("Parsed item involvements for %d events", len(item_involvements), extra={"chapter": 0})
        return item_involvements

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

            logger.info("Successfully created %d ActKeyEvent event nodes", len(act_events), extra={"chapter": self.chapter_number})

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

            logger.info("Successfully enriched %d Location nodes with names", len(location_names), extra={"chapter": self.chapter_number})

            return True

        except Exception as e:
            logger.error("Error enriching Location names: %s", str(e), exc_info=True)
            return False

    async def create_event_relationships(
        self,
        act_events: list[ActKeyEvent],
        character_involvements: dict[str, list[tuple[str, str | None]]],
        location_involvements: dict[str, str],
        item_involvements: dict[str, list[tuple[str, str]]] | None = None,
    ) -> bool:
        """Create relationships between events, characters, locations, and items.

        This method creates all relationship types defined in the schema design:
        - PART_OF relationships between ActKeyEvents and MajorPlotPoints
        - HAPPENS_BEFORE relationships between events
        - INVOLVES relationships between events and characters
        - OCCURS_AT relationships between events and locations
        - FEATURES_ITEM relationships between events and items

        Args:
            act_events: List of ActKeyEvent instances
            character_involvements: Dict mapping event IDs to character names with roles
            location_involvements: Dict mapping event IDs to location names
            item_involvements: Dict mapping event IDs to item names with roles (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            cypher_queries = []

            # Create PART_OF relationships between ActKeyEvents and MajorPlotPoints
            for act_event in act_events:
                if act_event.act_number == 1 and act_event.sequence_in_act <= 2:
                    major_plot_point_name = "Inciting Incident"
                elif act_event.act_number == 1 and act_event.sequence_in_act > 2:
                    major_plot_point_name = "Midpoint"
                elif act_event.act_number == 2:
                    major_plot_point_name = "Climax"
                else:
                    major_plot_point_name = "Resolution"

                query = """
                MATCH (major:Event {event_type: "MajorPlotPoint"})
                WHERE major.name CONTAINS $major_plot_point_name
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
            for i in range(len(act_events)):
                for j in range(i + 1, len(act_events)):
                    event_a = act_events[i]
                    event_b = act_events[j]

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
            involves_count = 0
            for event_id, characters in character_involvements.items():
                for character_name, role in characters:
                    query = """
                    MATCH (e:Event {id: $event_id})
                    MATCH (c:Character {name: $character_name})
                    MERGE (e)-[r:INVOLVES]->(c)
                    SET r.role = $role,
                        r.created_ts = timestamp(),
                        r.updated_ts = timestamp()
                    """

                    params = {
                        "event_id": event_id,
                        "character_name": character_name,
                        "role": role or "participant",
                    }

                    cypher_queries.append((query, params))
                    involves_count += 1

            # Create OCCURS_AT relationships with locations
            occurs_at_count = 0
            for event_id, location_identifier in location_involvements.items():
                query = """
                MATCH (e:Event {id: $event_id})
                MATCH (l:Location)
                WHERE l.name = $location_identifier OR l.description = $location_identifier
                MERGE (e)-[r:OCCURS_AT]->(l)
                SET r.created_ts = timestamp(),
                    r.updated_ts = timestamp()
                """

                params = {
                    "event_id": event_id,
                    "location_identifier": location_identifier,
                }

                cypher_queries.append((query, params))
                occurs_at_count += 1

            # Create FEATURES_ITEM relationships with items
            features_item_count = 0
            if item_involvements:
                for event_id, items in item_involvements.items():
                    for item_name, role in items:
                        query = """
                        MATCH (e:Event {id: $event_id})
                        MATCH (i:Item)
                        WHERE i.name = $item_name OR i.name CONTAINS $item_name OR $item_name CONTAINS i.name
                        MERGE (e)-[r:FEATURES_ITEM]->(i)
                        SET r.role = $role,
                            r.created_ts = timestamp(),
                            r.updated_ts = timestamp()
                        """

                        params = {
                            "event_id": event_id,
                            "item_name": item_name,
                            "role": role,
                        }

                        cypher_queries.append((query, params))
                        features_item_count += 1

            # Execute all queries
            for query, params in cypher_queries:
                await neo4j_manager.execute_write_query(query, params)

            logger.info(
                "Successfully created event relationships: PART_OF, HAPPENS_BEFORE, %d INVOLVES, %d OCCURS_AT, %d FEATURES_ITEM",
                involves_count,
                occurs_at_count,
                features_item_count,
                extra={"chapter": self.chapter_number},
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

            # Step 5b: Extract new locations and items from act narrative text
            logger.info("Extracting new world items from act outline text")
            new_locations, new_items = await self.extract_new_world_items(act_outline_data)

            if new_locations:
                await self.create_new_location_nodes(new_locations)
            if new_items:
                await self.create_new_item_nodes(new_items)

            # Step 6: Extract character involvements
            logger.info("Extracting character involvements from events")
            character_involvements = await self._parse_character_involvements(act_events)

            # Step 7: Extract location involvements
            logger.info("Extracting location involvements from events")
            location_involvements = await self._parse_location_involvements(act_events)

            # Step 8: Get all items for item involvement extraction
            logger.info("Fetching items for item involvement extraction")
            items = await self._get_all_items()

            # Step 9: Extract item involvements
            item_involvements = {}
            if items:
                logger.info("Extracting item involvements from events")
                item_involvements = await self._extract_event_item_involvements(act_events, items)

                if not item_involvements:
                    logger.info("No item involvements found in act outline")

                logger.info("Parsed item involvements for %d events", len(item_involvements), extra={"chapter": self.chapter_number})

            # Step 10: Create event relationships
            logger.info("Creating event relationships in Neo4j")
            relationships_success = await self.create_event_relationships(act_events, character_involvements, location_involvements, item_involvements)

            if not relationships_success:
                return False, "Failed to create event relationships"

            involves_count = sum(len(chars) for chars in character_involvements.values())
            occurs_at_count = len(location_involvements)
            features_item_count = sum(len(items) for items in item_involvements.values())

            return (
                True,
                f"Successfully parsed and persisted "
                f"{len(act_events)} ActKeyEvents, "
                f"{len(location_names)} Location name enrichments, "
                f"{len(new_locations)} new Locations, "
                f"{len(new_items)} new Items, "
                f"{involves_count} INVOLVES relationships, "
                f"{occurs_at_count} OCCURS_AT relationships, and "
                f"{features_item_count} FEATURES_ITEM relationships",
            )

        except Exception as e:
            logger.error("Error in parse_and_persist: %s", str(e), exc_info=True)
            return False, f"Error parsing and persisting act outline: {str(e)}"


__all__ = ["ActOutlineParser"]
