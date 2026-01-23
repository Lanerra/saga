# core/parsers/global_outline_parser.py
"""Parse global outline and create Stage 2 knowledge graph entities.

This module provides the GlobalOutlineParser class that:
1. Reads global outline from JSON files
2. Creates MajorPlotPoint Event nodes (4 per story)
3. Creates Location nodes (major locations without names)
4. Creates Item nodes
5. Enriches Character nodes with arc properties
6. Persists to Neo4j using the existing data access layer

Based on: docs/schema-design.md - Stage 2: Global Outline
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import structlog
from pydantic import BaseModel, Field

import config
from core.db_manager import neo4j_manager
from core.exceptions import DatabaseError
from core.llm_interface_refactored import llm_service
from models.kg_models import CharacterProfile, Location, WorldItem
from processing.entity_deduplication import generate_entity_id
from prompts.prompt_renderer import get_system_prompt, render_prompt
from utils.common import ensure_exact_keys, try_load_json_from_response

logger = structlog.get_logger(__name__)


class MajorPlotPoint(BaseModel):
    """Represents a major plot point in the story (Stage 2)."""

    id: str
    name: str
    description: str
    event_type: str = "MajorPlotPoint"
    sequence_order: int
    created_chapter: int = 0
    is_provisional: bool = False
    created_ts: int | None = None
    updated_ts: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MajorPlotPoint:
        """Create a MajorPlotPoint from a dictionary.

        Args:
            data: Dictionary containing plot point data

        Returns:
            MajorPlotPoint instance
        """
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            sequence_order=data.get("sequence_order", 0),
            created_chapter=data.get("created_chapter", 0),
            is_provisional=data.get("is_provisional", False),
            created_ts=data.get("created_ts"),
            updated_ts=data.get("updated_ts"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "event_type": self.event_type,
            "sequence_order": self.sequence_order,
            "created_chapter": self.created_chapter,
            "is_provisional": self.is_provisional,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
        }


class GlobalOutlineParser:
    """Parse global outline and create Stage 2 knowledge graph entities.

    This parser handles Stage 2 of the knowledge graph construction:
    - MajorPlotPoint Event node creation (4 per story)
    - Location node creation (major locations)
    - Item node creation
    - Character node enrichment with arc properties
    - Persistence to Neo4j

    Attributes:
        global_outline_path: Path to global outline JSON file
        chapter_number: Chapter number for provenance (0 for initialization)
    """

    def __init__(self, global_outline_path: str = "global_outline/main_v1.json", chapter_number: int = 0):
        """Initialize the GlobalOutlineParser.

        Args:
            global_outline_path: Path to global outline JSON file
            chapter_number: Chapter number for provenance (0 for initialization)
        """
        self.global_outline_path = global_outline_path
        self.chapter_number = chapter_number
        self._world_items_cache: list[WorldItem] | None = None

    async def parse_global_outline(self) -> dict[str, Any]:
        """Parse global outline from JSON file.

        Returns:
            Dictionary containing parsed global outline data

        Raises:
            ValueError: If global outline file cannot be read or parsed
            DatabaseError: If there are issues persisting to Neo4j
        """
        try:
            # Read the global outline JSON file
            with open(self.global_outline_path, 'r', encoding='utf-8') as f:
                global_outline_data = json.load(f)
        except FileNotFoundError as e:
            logger.error(f"Global outline file not found: {self.global_outline_path}", exc_info=True)
            raise ValueError(f"Global outline file not found: {self.global_outline_path}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in global outline file: {self.global_outline_path}", exc_info=True)
            raise ValueError(f"Invalid JSON in global outline file: {self.global_outline_path}") from e

        return global_outline_data

    def _generate_event_id(self, event_name: str, sequence_order: int) -> str:
        """Generate a stable ID for an event.

        Args:
            event_name: Name of the event
            sequence_order: Sequence order of the event (1-4)

        Returns:
            Generated event ID
        """
        # Use SHA256 hash of event name + sequence order for stable ID
        hash_input = f"{event_name}_{sequence_order}"
        hash_obj = hashlib.sha256(hash_input.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()[:16]  # Use first 16 chars for brevity
        return f"event_{hash_hex}"

    def _parse_major_plot_points(self, global_outline_data: dict[str, Any]) -> list[MajorPlotPoint]:
        """Parse major plot points from global outline data.

        Args:
            global_outline_data: Parsed global outline data

        Returns:
            List of MajorPlotPoint instances

        Raises:
            ValueError: If required plot points are missing or invalid
        """
        # Major plot points should be in the global outline
        # They are typically: inciting_incident, midpoint, climax, resolution
        plot_points = []

        # Define the expected sequence order for each plot point type
        plot_point_mapping = {
            "inciting_incident": 1,
            "midpoint": 2,
            "climax": 3,
            "resolution": 4,
        }

        for plot_point_type, sequence_order in plot_point_mapping.items():
            plot_point_name = global_outline_data.get(plot_point_type, "")
            if not plot_point_name:
                logger.warning(f"Missing plot point: {plot_point_type}")
                continue

            # Generate a stable ID for this plot point
            event_id = self._generate_event_id(plot_point_name, sequence_order)

            # Create MajorPlotPoint
            plot_point = MajorPlotPoint(
                id=event_id,
                name=plot_point_name,
                description=global_outline_data.get(plot_point_type, ""),
                sequence_order=sequence_order,
                created_chapter=0,
                is_provisional=False,
                created_ts=global_outline_data.get("created_ts"),
                updated_ts=global_outline_data.get("updated_ts"),
            )

            plot_points.append(plot_point)

        # Validate we have exactly 4 major plot points
        if len(plot_points) != 4:
            logger.error(f"Expected 4 major plot points, got {len(plot_points)}")
            raise ValueError(f"Expected 4 major plot points, got {len(plot_points)}")

        return plot_points

    async def _parse_locations(self, global_outline_data: dict[str, Any]) -> list[Location]:
        """Parse locations from global outline data.

        Extracts locations from narrative text using LLM-based extraction.

        Args:
            global_outline_data: Parsed global outline data

        Returns:
            List of Location instances
        """
        if self._world_items_cache is None:
            self._world_items_cache = await self._extract_world_items_from_outline(global_outline_data)

        locations = []
        for item in self._world_items_cache:
            if item.category == "location":
                location = Location(
                    id=item.id,
                    name=None,
                    description=item.description,
                    category="Location",
                    created_chapter=0,
                    is_provisional=False,
                    created_ts=item.created_ts,
                    updated_ts=item.updated_ts,
                )
                locations.append(location)

        return locations

    async def _parse_items(self, global_outline_data: dict[str, Any]) -> list[WorldItem]:
        """Parse items from global outline data.

        Extracts items (objects) from narrative text using LLM-based extraction.

        Args:
            global_outline_data: Parsed global outline data

        Returns:
            List of WorldItem instances
        """
        if self._world_items_cache is None:
            self._world_items_cache = await self._extract_world_items_from_outline(global_outline_data)

        items = []
        for item in self._world_items_cache:
            if item.category == "object":
                items.append(item)

        return items

    def _parse_character_arcs(self, global_outline_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Parse character arcs from global outline data.

        Args:
            global_outline_data: Parsed global outline data

        Returns:
            Dictionary mapping character names to their arc data
        """
        character_arcs = {}

        if "character_arcs" in global_outline_data:
            for arc_data in global_outline_data["character_arcs"]:
                character_name = arc_data.get("character_name", "")
                if not character_name:
                    continue

                character_arcs[character_name] = {
                    "arc_start": arc_data.get("starting_state", ""),
                    "arc_end": arc_data.get("ending_state", ""),
                    "arc_key_moments": arc_data.get("key_moments", []),
                }

        return character_arcs

    async def _extract_world_items_from_outline(self, global_outline_data: dict[str, Any]) -> list[WorldItem]:
        """Extract world items (locations and objects) from global outline using LLM.

        This method extracts locations and items from the narrative text in the global outline
        using LLM-based entity extraction.

        Args:
            global_outline_data: Parsed global outline data

        Returns:
            List of WorldItem instances (with category="location" or category="object")
        """
        outline_text = global_outline_data.get("raw_text", "")
        if not outline_text:
            outline_text = json.dumps(global_outline_data, indent=2)

        prompt = render_prompt(
            "knowledge_agent/extract_world_items_lines.j2",
            {
                "setting": global_outline_data.get("setting", ""),
                "outline_text": outline_text,
            },
        )

        for attempt in range(1, 3):
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

                return self._parse_world_items_extraction(response)
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == 2:
                    logger.warning(
                        "Failed to extract world items after %d attempts: %s",
                        attempt,
                        str(e),
                        exc_info=True
                    )
                    return []

        return []

    def _parse_world_items_extraction(self, response: str) -> list[WorldItem]:
        """Parse LLM response into WorldItem models.

        Args:
            response: LLM response (JSON array)

        Returns:
            List of WorldItem models

        Raises:
            ValueError: If the output violates the JSON/schema contract
        """
        raw_text = response.strip()

        data, candidates_tried, parse_errors = try_load_json_from_response(raw_text)

        if data is None:
            raise ValueError(f"Failed to parse JSON from response. Errors: {parse_errors}")

        if not isinstance(data, list):
            raise ValueError("World items extraction must be a JSON array")

        items: list[WorldItem] = []
        allowed_categories = {"location", "object"}

        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"World item at index {index} must be a JSON object")

            required_keys = {"name", "category", "description"}
            missing_keys = required_keys - set(item.keys())
            if missing_keys:
                logger.warning(
                    "World item at index %d missing required keys: %s",
                    index,
                    missing_keys
                )
                continue

            name = item["name"]
            category = item["category"]
            description = item["description"]

            if not isinstance(name, str) or not name.strip():
                logger.warning("World item at index %d has invalid name", index)
                continue
            if not isinstance(category, str) or category not in allowed_categories:
                logger.warning(
                    "World item at index %d has invalid category: %s (expected one of %s)",
                    index,
                    category,
                    sorted(allowed_categories)
                )
                continue
            if not isinstance(description, str) or not description.strip():
                logger.warning("World item at index %d has invalid description", index)
                continue

            items.append(
                WorldItem(
                    id=generate_entity_id(name.strip(), category, chapter=0),
                    name=name.strip(),
                    description=description.strip(),
                    category=category,
                    created_chapter=0,
                    is_provisional=False,
                )
            )

        logger.info(
            "_parse_world_items_extraction: extracted world items",
            count=len(items),
        )

        return items

    async def create_major_plot_point_nodes(self, plot_points: list[MajorPlotPoint]) -> bool:
        """Create MajorPlotPoint Event nodes in Neo4j.

        Args:
            plot_points: List of MajorPlotPoint instances to create

        Returns:
            True if successful, False otherwise
        """
        try:
            # Build Cypher query for creating event nodes
            cypher_queries = []

            for plot_point in plot_points:
                query = """
                MERGE (e:Event {id: $id})
                ON CREATE SET 
                    e.name = $name,
                    e.description = $description,
                    e.event_type = $event_type,
                    e.sequence_order = $sequence_order,
                    e.created_chapter = $created_chapter,
                    e.is_provisional = $is_provisional,
                    e.created_ts = timestamp(),
                    e.updated_ts = timestamp()
                ON MATCH SET 
                    e.name = $name,
                    e.description = $description,
                    e.event_type = $event_type,
                    e.sequence_order = $sequence_order,
                    e.created_chapter = $created_chapter,
                    e.is_provisional = $is_provisional,
                    e.updated_ts = timestamp()
                """

                params = {
                    "id": plot_point.id,
                    "name": plot_point.name,
                    "description": plot_point.description,
                    "event_type": plot_point.event_type,
                    "sequence_order": plot_point.sequence_order,
                    "created_chapter": plot_point.created_chapter,
                    "is_provisional": plot_point.is_provisional,
                }

                cypher_queries.append((query, params))

            # Execute all queries
            for query, params in cypher_queries:
                await neo4j_manager.execute_write_query(query, params)

            logger.info(
                "Successfully created %d MajorPlotPoint event nodes",
                len(plot_points),
                extra={"chapter": self.chapter_number}
            )

            return True

        except Exception as e:
            logger.error("Error creating MajorPlotPoint nodes: %s", str(e), exc_info=True)
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

    async def create_item_nodes(self, items: list[WorldItem]) -> bool:
        """Create Item nodes in Neo4j.

        Args:
            items: List of WorldItem instances to create

        Returns:
            True if successful, False otherwise
        """
        try:
            # Build Cypher query for creating item nodes
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
                ON MATCH SET 
                    i.name = $name,
                    i.description = $description,
                    i.category = $category,
                    i.created_chapter = $created_chapter,
                    i.is_provisional = $is_provisional,
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

            # Execute all queries
            for query, params in cypher_queries:
                await neo4j_manager.execute_write_query(query, params)

            logger.info(
                "Successfully created %d Item nodes",
                len(items),
                extra={"chapter": self.chapter_number}
            )

            return True

        except Exception as e:
            logger.error("Error creating Item nodes: %s", str(e), exc_info=True)
            return False

    async def enrich_character_arcs(self, character_arcs: dict[str, dict[str, Any]]) -> bool:
        """Enrich Character nodes with arc properties.

        Args:
            character_arcs: Dictionary mapping character names to arc data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Build Cypher query for updating character arcs
            cypher_queries = []

            for character_name, arc_data in character_arcs.items():
                query = """
                MATCH (c:Character {name: $character_name})
                SET c.arc_start = $arc_start,
                    c.arc_end = $arc_end,
                    c.arc_key_moments = $arc_key_moments,
                    c.updated_ts = timestamp()
                """

                params = {
                    "character_name": character_name,
                    "arc_start": arc_data.get("arc_start", ""),
                    "arc_end": arc_data.get("arc_end", ""),
                    "arc_key_moments": arc_data.get("arc_key_moments", []),
                }

                cypher_queries.append((query, params))

            # Execute all queries
            for query, params in cypher_queries:
                await neo4j_manager.execute_write_query(query, params)

            logger.info(
                "Successfully enriched %d Character nodes with arc properties",
                len(character_arcs),
                extra={"chapter": self.chapter_number}
            )

            return True

        except Exception as e:
            logger.error("Error enriching Character arcs: %s", str(e), exc_info=True)
            return False

    async def parse_and_persist(self) -> tuple[bool, str]:
        """Parse global outline and persist to Neo4j.

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Step 1: Parse global outline
            logger.info("Parsing global outline from %s", self.global_outline_path)
            global_outline_data = await self.parse_global_outline()

            if not global_outline_data:
                return False, "No data found in global outline"

            logger.info("Parsed global outline data", extra={"chapter": self.chapter_number})

            # Step 2: Parse major plot points
            logger.info("Parsing major plot points from global outline")
            plot_points = self._parse_major_plot_points(global_outline_data)

            if not plot_points:
                return False, "No major plot points found in global outline"

            logger.info("Parsed %d major plot points", len(plot_points), extra={"chapter": self.chapter_number})

            # Step 3: Parse locations
            logger.info("Parsing locations from global outline")
            locations = await self._parse_locations(global_outline_data)

            if not locations:
                logger.warning("No locations found in global outline")

            logger.info("Parsed %d locations", len(locations), extra={"chapter": self.chapter_number})

            # Step 4: Parse items
            logger.info("Parsing items from global outline")
            items = await self._parse_items(global_outline_data)

            if not items:
                logger.warning("No items found in global outline")

            logger.info("Parsed %d items", len(items), extra={"chapter": self.chapter_number})

            # Step 5: Parse character arcs
            logger.info("Parsing character arcs from global outline")
            character_arcs = self._parse_character_arcs(global_outline_data)

            if not character_arcs:
                logger.warning("No character arcs found in global outline")

            logger.info("Parsed %d character arcs", len(character_arcs), extra={"chapter": self.chapter_number})

            # Step 6: Create major plot point nodes
            logger.info("Creating MajorPlotPoint event nodes in Neo4j")
            plot_points_success = await self.create_major_plot_point_nodes(plot_points)

            if not plot_points_success:
                return False, "Failed to create MajorPlotPoint nodes"

            # Step 7: Create location nodes
            logger.info("Creating Location nodes in Neo4j")
            locations_success = await self.create_location_nodes(locations)

            if not locations_success:
                return False, "Failed to create Location nodes"

            # Step 8: Create item nodes
            logger.info("Creating Item nodes in Neo4j")
            items_success = await self.create_item_nodes(items)

            if not items_success:
                return False, "Failed to create Item nodes"

            # Step 9: Enrich character arcs
            logger.info("Enriching Character nodes with arc properties")
            character_arcs_success = await self.enrich_character_arcs(character_arcs)

            if not character_arcs_success:
                return False, "Failed to enrich Character arcs"

            return (
                True,
                f"Successfully parsed and persisted "
                f"{len(plot_points)} MajorPlotPoints, "
                f"{len(locations)} Locations, "
                f"{len(items)} Items, and "
                f"{len(character_arcs)} Character arcs"
            )

        except Exception as e:
            logger.error("Error in parse_and_persist: %s", str(e), exc_info=True)
            return False, f"Error parsing and persisting global outline: {str(e)}"


__all__ = ["GlobalOutlineParser", "MajorPlotPoint"]
