# core/parsers/character_sheet_parser.py
"""Parse character sheets and create Character nodes and relationships.

This module provides the CharacterSheetParser class that:
1. Reads character sheets from JSON files
2. Creates Character nodes with all Stage 1 properties
3. Creates Character-Character relationships
4. Persists to Neo4j using the existing data access layer

Based on: docs/schema-design.md - Stage 1: Character Initialization
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from core.db_manager import neo4j_manager
from data_access.character_queries import sync_characters
from models.kg_models import CharacterProfile
from utils.text_processing import validate_and_filter_traits

logger = structlog.get_logger(__name__)


class CharacterSheetParser:
    """Parse character sheets and create Character nodes and relationships.

    This parser handles Stage 1 of the knowledge graph construction:
    - Character node creation with all Stage 1 properties
    - Character-Character relationship creation
    - Persistence to Neo4j

    Attributes:
        character_sheets_path: Path to character sheets JSON file
        chapter_number: Chapter number for provenance (0 for initialization)
    """

    def __init__(self, character_sheets_path: str = "character_sheets/all_v1.json", chapter_number: int = 0):
        """Initialize the CharacterSheetParser.

        Args:
            character_sheets_path: Path to character sheets JSON file
            chapter_number: Chapter number for provenance (0 for initialization)
        """
        self.character_sheets_path = character_sheets_path
        self.chapter_number = chapter_number

    async def parse_character_sheets(self) -> list[CharacterProfile]:
        """Parse character sheets from JSON file.

        Returns:
            List of CharacterProfile objects

        Raises:
            ValueError: If character sheets file cannot be read or parsed
            DatabaseError: If there are issues persisting to Neo4j
        """
        try:
            # Read the character sheets JSON file
            with open(self.character_sheets_path, encoding="utf-8") as f:
                character_sheets_data = json.load(f)
        except FileNotFoundError as e:
            logger.error(f"Character sheets file not found: {self.character_sheets_path}", exc_info=True)
            raise ValueError(f"Character sheets file not found: {self.character_sheets_path}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in character sheets file: {self.character_sheets_path}", exc_info=True)
            raise ValueError(f"Invalid JSON in character sheets file: {self.character_sheets_path}") from e

        # Parse each character sheet
        characters = []
        for character_name, character_data in character_sheets_data.items():
            try:
                character_profile = self._parse_character_sheet(character_name, character_data)
                characters.append(character_profile)
            except Exception as e:
                logger.error(f"Error parsing character sheet for {character_name}: {str(e)}", exc_info=True)
                raise ValueError(f"Error parsing character sheet for {character_name}: {str(e)}") from e

        return characters

    def _parse_character_sheet(self, character_name: str, character_data: dict[str, Any]) -> CharacterProfile:
        """Parse a single character sheet into a CharacterProfile.

        Args:
            character_name: Name of the character
            character_data: Dictionary containing character data

        Returns:
            CharacterProfile object

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate required fields
        if not character_data.get("name"):
            character_data["name"] = character_name

        if not character_data.get("description"):
            raise ValueError(f"Missing 'description' field for character: {character_name}")

        if not character_data.get("traits"):
            character_data["traits"] = []

        if not character_data.get("status"):
            character_data["status"] = "Active"

        allowed_statuses = ["Active", "Dead", "Injured", "Unknown", "Missing"]
        if character_data["status"] not in allowed_statuses:
            raise ValueError(f"Invalid status {character_data['status']!r} for character {character_name!r}. " f"Allowed values: {allowed_statuses}")

        # Validate and filter traits to ensure single-word format
        raw_traits = character_data.get("traits", [])
        if not isinstance(raw_traits, list):
            raw_traits = []

        # Filter to only string traits
        string_traits = [t for t in raw_traits if isinstance(t, str)]

        # Validate and filter traits
        validated_traits = validate_and_filter_traits(string_traits)

        if len(validated_traits) != len(string_traits):
            logger.warning(
                "Filtered invalid traits for character",
                character=character_name,
                original_count=len(string_traits),
                filtered_count=len(validated_traits),
                removed=set(string_traits) - set(validated_traits),
            )

        # Parse relationships from the character data
        relationships = {}
        if "relationships" in character_data:
            for target_name, rel_data in character_data["relationships"].items():
                if not isinstance(rel_data, dict):
                    raise ValueError(f"Relationship data for {character_name} -> {target_name} must be a dict " f"with 'type' and 'description' keys, got {type(rel_data).__name__}")
                relationships[target_name] = rel_data

        # Create CharacterProfile
        character_profile = CharacterProfile(
            name=character_data["name"],
            personality_description=character_data.get("description", ""),
            traits=validated_traits,
            status=character_data["status"],
            created_chapter=self.chapter_number,
            is_provisional=False,  # Stage 1 characters are not provisional
            created_ts=None,  # Will be set by Neo4j
            updated_ts=None,  # Will be set by Neo4j
            arc_start=None,  # Stage 2 property
            arc_end=None,  # Stage 2 property
            arc_key_moments=[],  # Stage 2 property
            physical_description=None,  # Stage 5 property
            relationships=relationships,
        )

        return character_profile

    def _normalize_character_name(self, name: str) -> str:
        """Normalize character name by removing titles and honorifics.

        Args:
            name: Character name to normalize

        Returns:
            Normalized name without titles
        """
        if not name:
            return ""

        titles = [
            "Dr.",
            "Dr",
            "Doctor",
            "Prof.",
            "Prof",
            "Professor",
            "Mr.",
            "Mr",
            "Mister",
            "Mrs.",
            "Mrs",
            "Missus",
            "Ms.",
            "Ms",
            "Miss",
            "Director",
            "Captain",
            "Capt.",
            "Capt",
            "Lieutenant",
            "Lt.",
            "Lt",
            "Sergeant",
            "Sgt.",
            "Sgt",
            "Major",
            "Maj.",
            "Maj",
            "Colonel",
            "Col.",
            "Col",
            "General",
            "Gen.",
            "Gen",
            "Admiral",
            "Adm.",
            "Adm",
            "Commander",
            "Cmdr.",
            "Cmdr",
            "Lord",
            "Lady",
            "Sir",
            "Dame",
            "Reverend",
            "Rev.",
            "Rev",
            "Father",
            "Mother",
            "Brother",
            "Sister",
        ]

        normalized = name.strip()

        for title in titles:
            if normalized.startswith(title + " "):
                normalized = normalized[len(title) :].strip()
                break

        return normalized

    def _find_character_by_name(self, target_name: str, character_map: dict) -> str | None:
        """Find a character in the map, trying various normalization strategies.

        Args:
            target_name: Name to search for
            character_map: Dictionary of character names to character data

        Returns:
            Matching character name from the map, or None if not found
        """
        if target_name in character_map:
            return target_name

        target_normalized = self._normalize_character_name(target_name)

        for char_name in character_map.keys():
            if char_name == target_name:
                return char_name

            char_normalized = self._normalize_character_name(char_name)

            if char_normalized.lower() == target_normalized.lower():
                return char_name

            if target_normalized.lower() in char_normalized.lower() or char_normalized.lower() in target_normalized.lower():
                if len(target_normalized) > 3 and len(char_normalized) > 3:
                    return char_name

        return None

    async def parse_relationships(self, characters: list[CharacterProfile]) -> dict[str, dict[str, Any]]:
        """Parse relationships from character sheets.

        Args:
            characters: List of CharacterProfile objects

        Returns:
            Dictionary mapping character names to their relationships
        """
        character_map = {char.name: char for char in characters}

        relationships = {}

        for character_name, character_data in character_map.items():
            if hasattr(character_data, "relationships") and character_data.relationships:
                for target_name, rel_data in character_data.relationships.items():
                    matched_name = self._find_character_by_name(target_name, character_map)

                    if matched_name is None:
                        logger.warning(f"Character {character_name} references non-existent character {target_name} in relationship")
                        continue

                    if matched_name != target_name:
                        logger.debug(f"Fuzzy matched character reference: '{target_name}' -> '{matched_name}'", source=character_name)

                    # Validate relationship type
                    if not isinstance(rel_data, dict):
                        logger.warning(f"Invalid relationship data for {character_name} -> {matched_name}")
                        continue

                    rel_type = rel_data.get("type", "")
                    if not rel_type:
                        logger.warning(f"Missing relationship type for {character_name} -> {matched_name}")
                        continue

                    if character_name not in relationships:
                        relationships[character_name] = {}

                    relationships[character_name][matched_name] = {
                        "type": rel_type,
                        "description": rel_data.get("description", ""),
                        "chapter_added": self.chapter_number,
                        "source_profile_managed": True,
                    }

        return relationships

    async def create_character_nodes(self, characters: list[CharacterProfile]) -> bool:
        """Create Character nodes in Neo4j.

        Args:
            characters: List of CharacterProfile objects to create

        Returns:
            True if successful, False otherwise
        """
        try:
            # Use the existing sync_characters function to persist characters
            success = await sync_characters(characters, self.chapter_number)

            if not success:
                logger.error("Failed to persist character nodes to Neo4j")
                return False

            logger.info("Successfully created %d character nodes", len(characters), extra={"chapter": self.chapter_number})

            return True

        except Exception as e:
            logger.error("Error creating character nodes: %s", str(e), exc_info=True)
            return False

    async def create_relationships(self, relationships: dict[str, dict[str, Any]]) -> bool:
        """Create Character-Character relationships in Neo4j.

        Args:
            relationships: Dictionary of relationships to create

        Returns:
            True if successful, False otherwise
        """
        try:
            # Build Cypher query for relationship creation
            cypher_queries = []

            for source_name, targets in relationships.items():
                for target_name, rel_data in targets.items():
                    rel_type = rel_data["type"]
                    description = rel_data.get("description", "")
                    chapter_added = rel_data.get("chapter_added", 0)

                    # Create relationship query
                    query = f"""
                    MATCH (source:Character {{name: $source_name}})
                    MATCH (target:Character {{name: $target_name}})
                    MERGE (source)-[r:{rel_type}]->(target)
                    SET r.description = $description,
                        r.chapter_added = $chapter_added,
                        r.source_profile_managed = $source_profile_managed,
                        r.created_ts = timestamp(),
                        r.updated_ts = timestamp()
                    """

                    params = {
                        "source_name": source_name,
                        "target_name": target_name,
                        "description": description,
                        "chapter_added": chapter_added,
                        "source_profile_managed": True,
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

    async def parse_and_persist(self) -> tuple[bool, str]:
        """Parse character sheets and persist to Neo4j.

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Step 1: Parse character sheets
            logger.info("Parsing character sheets from %s", self.character_sheets_path)
            characters = await self.parse_character_sheets()

            if not characters:
                return False, "No characters found in character sheets"

            logger.info("Parsed %d characters from character sheets", len(characters))

            # Step 2: Parse relationships
            logger.info("Parsing relationships from character sheets")
            relationships = await self.parse_relationships(characters)

            if not relationships:
                logger.warning("No relationships found in character sheets")

            logger.info("Parsed %d relationships from character sheets", len(relationships))

            # Step 3: Create character nodes
            logger.info("Creating character nodes in Neo4j")
            nodes_success = await self.create_character_nodes(characters)

            if not nodes_success:
                return False, "Failed to create character nodes"

            # Step 4: Create relationships
            logger.info("Creating relationships in Neo4j")
            rels_success = await self.create_relationships(relationships)

            if not rels_success:
                return False, "Failed to create relationships"

            return True, f"Successfully parsed and persisted {len(characters)} characters and {len(relationships)} relationships"

        except Exception as e:
            logger.error("Error in parse_and_persist: %s", str(e), exc_info=True)
            return False, f"Error parsing and persisting character sheets: {str(e)}"


__all__ = ["CharacterSheetParser"]
